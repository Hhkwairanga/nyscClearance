"""
Accounts app views

Key areas:
- Authentication and profile management (register, login, profile, stats)
- Organization structure (branches, departments, units, corpers)
- Attendance capture with geofence and simple face detection
- Performance clearance generation and verification
- Wallets and transactions (ORG/BRANCH/CORPER) with deduction order:
  ORG -> BRANCH -> CORPER; welcome bonus applies to ORG only
- System settings, announcements for org dashboards
- Paystack funding: initialize and verify using keys stored in DB

Notes:
- Uses timezone-aware `timezone.localtime()` for logging and `Africa/Lagos` by default
- Avoids double charges by checking reference across wallets
"""

from django.contrib.auth import get_user_model, authenticate, login, logout
from django.conf import settings
from django.shortcuts import redirect
from rest_framework import status, permissions, viewsets
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import action
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.core.cache import cache
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponseNotFound
from django.utils import timezone
from django.utils.decorators import method_decorator
import requests
import base64
import math
import numpy as np
import cv2
import os
import shutil
import uuid
import traceback
import face_recognition

from .serializers import (
    OrganizationRegisterSerializer,
    LoginSerializer,
    OrganizationProfileSerializer,
    BranchOfficeSerializer,
    DepartmentSerializer,
    UnitSerializer,
    CorpMemberSerializer,
    PublicHolidaySerializer,
    LeaveRequestSerializer,
    NotificationSerializer,
)
from .tokens import validate_email_token, generate_email_token
from .models import OrganizationProfile, BranchOffice, Department, Unit, CorpMember, PublicHoliday, LeaveRequest, Notification, AttendanceLog, WalletAccount, WalletTransaction, ClearanceOverride, TempFaceEncoding
from django.db.models import Count
from django.db import models


User = get_user_model()

# In-memory capture state per corper_id
_CAPTURE_STATE = {}
_ATTENDANCE_STATE = {}

def _ensure_rgb_uint8(arr):
    """Ensure image array is uint8 RGB (H, W, 3).
    - Cast dtype to uint8 (with clipping).
    - Drop alpha channel if present.
    - Expand/convert grayscale to 3 channels.
    """
    if arr is None:
        return None
    try:
        # Ensure dtype uint8; scale floats in 0..1 range if needed
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(arr) or 0.0) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Ensure 3 channels
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            elif arr.shape[2] == 3:
                pass
            else:
                # Unexpected channel count; best effort: take first 3 or tile
                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                else:
                    arr = np.repeat(arr, 3, axis=2)[:, :, :3]
        return arr
    except Exception as e:
        print("[capture] _ensure_rgb_uint8 error:", e)
        traceback.print_exc()
        return arr

def sanitize_rgb(img):
    """Convert a decoded frame to uint8 3-channel RGB.
    - Accepts BGR/GRAY/BGRA inputs, drops alpha if present.
    - Casts dtype to uint8 with clipping/scaling when necessary.
    """
    if img is None:
        return None
    try:
        arr = img
        # Type normalize first
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                maxv = float(np.nanmax(arr)) if arr.size else 1.0
                scale = 255.0 if maxv <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Channel handling and BGR->RGB conversion
        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            ch = arr.shape[2]
            if ch == 4:
                arr = arr[:, :, :3]
            if arr.shape[2] >= 3:
                rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = None
        rgb = _ensure_rgb_uint8(rgb) if rgb is not None else None
        if rgb is None:
            return None
        if not rgb.flags['C_CONTIGUOUS']:
            rgb = np.ascontiguousarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        return rgb
    except Exception as e:
        print("[capture] sanitize_rgb error:", e)
        traceback.print_exc()
        return None

def ensure_dlib_rgb(arr):
    """Return a new C-contiguous uint8 RGB array suitable for dlib.
    Always makes a copy to avoid any exotic strides/views that dlib rejects.
    """
    if arr is None:
        return None
    try:
        a = arr
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        elif a.ndim == 3:
            if a.shape[2] == 4:
                a = a[:, :, :3]
            elif a.shape[2] < 3:
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        else:
            return None
        # Enforce dtype and memory layout with a copy
        a = np.array(a, dtype=np.uint8, order='C', copy=True)
        return a
    except Exception as e:
        print("[capture] ensure_dlib_rgb error:", e)
        traceback.print_exc()
        return None

def _safe_face_locations(img_rgb_or_gray, tag=""):
    """Call face_recognition.face_locations safely on a valid uint8 image.
    - First try on the provided image (expected RGB from sanitize_rgb).
    - If dlib complains about image type, fall back to grayscale detection.
    """
    try:
        locs = face_recognition.face_locations(img_rgb_or_gray)
        if not locs:
            # Try a higher upsample for small frames
            locs = face_recognition.face_locations(img_rgb_or_gray, number_of_times_to_upsample=2)
        if locs:
            return locs
        # If empty (no exception), try grayscale HOG as a second strategy
        arr = img_rgb_or_gray
        try:
            if arr is None:
                return []
            if arr.ndim == 3 and arr.shape[2] == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif arr.ndim == 2:
                gray = arr
            else:
                return []
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            if not gray.flags['C_CONTIGUOUS']:
                gray = np.ascontiguousarray(gray)
            _debug_img(f'{tag}.gray_noerr', gray)
            locs = face_recognition.face_locations(gray)
            if not locs:
                locs = face_recognition.face_locations(gray, number_of_times_to_upsample=2)
            if locs:
                return locs
        except Exception as ee_g:
            print(f"[detect] grayscale no-error path failed on {tag}: {ee_g}")
            traceback.print_exc()
        # Final fallback: Haar cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            locs = []
            for (x, y, w, h) in faces:
                top = int(y)
                left = int(x)
                bottom = int(y + h)
                right = int(x + w)
                locs.append((top, right, bottom, left))
            print(f"[detect] Haar fallback (no-error) produced {len(locs)} faces for {tag}")
            return locs
        except Exception as ee3:
            print(f"[detect] Haar fallback (no-error) failed on {tag}: {ee3}")
            traceback.print_exc()
            return []
    except RuntimeError as e:
        msg = str(e)
        print(f"[detect] face_locations failed on {tag}: {msg}")
        traceback.print_exc()
        # Fallback to grayscale if possible
        try:
            arr = img_rgb_or_gray
            if arr is None:
                return []
            if arr.ndim == 3 and arr.shape[2] == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif arr.ndim == 2:
                gray = arr
            else:
                return []
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            if not gray.flags['C_CONTIGUOUS']:
                gray = np.ascontiguousarray(gray)
            _debug_img(f'{tag}.gray', gray)
            try:
                locs = face_recognition.face_locations(gray)
                if not locs:
                    locs = face_recognition.face_locations(gray, number_of_times_to_upsample=2)
                if not locs:
                    print(f"[detect] No faces with HOG on gray (upsample<=2) for {tag}")
                return locs
            except RuntimeError as ee2:
                print(f"[detect] face_locations failed on grayscale {tag}: {ee2}")
                traceback.print_exc()
                # Final fallback: Haar cascade via OpenCV
                try:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    locs = []
                    for (x, y, w, h) in faces:
                        top = int(y)
                        left = int(x)
                        bottom = int(y + h)
                        right = int(x + w)
                        locs.append((top, right, bottom, left))
                    print(f"[detect] Haar fallback produced {len(locs)} faces for {tag}")
                    return locs
                except Exception as ee3:
                    print(f"[detect] Haar fallback failed on {tag}: {ee3}")
                    traceback.print_exc()
                    return []
        except Exception as ee:
            print(f"[detect] grayscale fallback failed on {tag}: {ee}")
            traceback.print_exc()
            return []
    except Exception as e:
        print(f"[detect] face_locations unexpected error on {tag}: {e}")
        traceback.print_exc()
        return []
    

def _reset_capture_state(corper_id: int):
    _CAPTURE_STATE[corper_id] = {
        'enc_count': 0,   # number of encodings accumulated
        'enc_sum': None,  # running sum of encoding vectors (numpy array)
    }

def _reset_attendance_state(corper_id: int):
    _ATTENDANCE_STATE[corper_id] = {
        'hits': 0,
        'logged': False,
    }

def _debug_img(tag, arr):
    try:
        if arr is None:
            print(f"[debug] {tag}: arr=None")
            return
        print(f"[debug] {tag}: dtype={arr.dtype}, shape={getattr(arr, 'shape', None)}, contig={arr.flags['C_CONTIGUOUS']}, strides={arr.strides}")
        try:
            amin = float(np.min(arr))
            amax = float(np.max(arr))
            print(f"[debug] {tag}: min={amin} max={amax}")
        except Exception:
            pass
    except Exception as e:
        print(f"[debug] {tag}: error printing info: {e}")
        traceback.print_exc()

def _process_capture_frame(corper_id: int, b64_frame: str, save_dir: str):
    # save_dir no longer used; kept for signature compatibility
    st = _CAPTURE_STATE.setdefault(corper_id, {'enc_count': 0, 'enc_sum': None})
    img_data = base64.b64decode(b64_frame)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, st['enc_count']

    # Sanitize to contiguous uint8 RGB for face_recognition
    rgb = sanitize_rgb(img)
    # Detect faces and compute first encoding
    try:
        locations = _safe_face_locations(rgb, tag='capture_inmem')
        img_for_enc = ensure_dlib_rgb(rgb)
        _debug_img('capture_inmem.enc_img', img_for_enc)
        encs = face_recognition.face_encodings(img_for_enc, locations) if img_for_enc is not None else []
    except Exception as e:
        print("[capture] Error in face_encodings (_process_capture_frame):", e)
        traceback.print_exc()
        locations, encs = [], []

    # If we have an encoding and haven't reached 30 yet, update running sum
    MAX_ENCODINGS = 30
    if encs and st.get('enc_count', 0) < MAX_ENCODINGS:
        vec = np.asarray(encs[0], dtype=np.float32)
        if st.get('enc_sum') is None:
            st['enc_sum'] = vec.copy()
        else:
            st['enc_sum'] = st['enc_sum'] + vec
        st['enc_count'] = st.get('enc_count', 0) + 1

    # draw overlays (use first face location if present)
    if locations:
        top, right, bottom, left = locations[0]
        cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(
            img,
            f'Enc {st.get("enc_count",0)}/{MAX_ENCODINGS}',
            (left, max(0, top-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2,
            cv2.LINE_AA
        )
    _, buffer = cv2.imencode('.jpg', img)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    return processed_frame, st.get('enc_count', 0)


def capture_page(request, corper_id: int):
    # Authorization: ORG or BRANCH can access; corper can also view own page
    user = request.user
    try:
        corper = CorpMember.objects.get(pk=corper_id)
    except CorpMember.DoesNotExist:
        return HttpResponseNotFound('Corper not found')

    # Basic auth checks
    allowed = False
    if getattr(user, 'role', None) == 'ORG' and getattr(corper.user, 'id', None) == user.id:
        allowed = True
    if getattr(user, 'role', None) == 'BRANCH':
        b = BranchOffice.objects.filter(admin=user).first()
        if b and b.id == getattr(corper.branch, 'id', None):
            allowed = True
    if getattr(user, 'role', None) == 'CORPER' and getattr(corper.account, 'id', None) == user.id:
        allowed = True
    if not allowed:
        raise PermissionDenied('Not allowed')

    _reset_capture_state(corper_id)
    # Prefer request Origin if it matches configured FRONTEND_ORIGINS
    request_origin = request.headers.get('Origin')
    try:
        allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
    except Exception:
        allowed = set()
    frontend_base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', 'http://localhost:5173'))).rstrip('/')

    # Ensure CSRF cookie is set for subsequent POSTs from the page
    try:
        get_token(request)
    except Exception:
        pass

    # Generate a unique session id for this capture flow
    session_id = uuid.uuid4().hex

    ctx = {
        'corper_id': corper.id,
        'full_name': corper.full_name,
        'state_code': corper.state_code,
        'frontend_base': frontend_base,
        'session_id': session_id,
    }
    return render(request, 'capture.html', ctx)


def capture_process_frame(request, corper_id: int):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    frame = request.POST.get('frame')
    session_id = request.POST.get('session')
    if not frame:
        return JsonResponse({'detail': 'Missing frame'}, status=400)
    if not session_id:
        return JsonResponse({'detail': 'Missing session'}, status=400)
    # Save dir per corper under media/captures/{id}
    media_root = getattr(settings, 'MEDIA_ROOT', os.path.join(os.getcwd(), 'media'))
    save_dir = os.path.join(media_root, 'captures', str(corper_id))
    processed, _ = _process_capture_frame(corper_id, frame, save_dir)
    if not processed:
        return JsonResponse({'detail': 'Processing failed'}, status=500)

    # Compute encoding for persistence (stateless across workers)
    try:
        img_data = base64.b64decode(frame)
        nparr = np.frombuffer(img_data, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = sanitize_rgb(bgr)
        _debug_img('capture_process_frame.rgb', rgb)
        locs = _safe_face_locations(rgb, tag='capture_persist') if rgb is not None else []
        img_for_enc = ensure_dlib_rgb(rgb) if rgb is not None else None
        _debug_img('capture_persist.enc_img', img_for_enc)
        encs = face_recognition.face_encodings(img_for_enc, locs) if img_for_enc is not None else []
    except Exception as e:
        print("[capture] Error in face_locations/encodings (persist step):", e)
        traceback.print_exc()
        encs = []

    MAX_ENCODINGS = 30
    current_count = TempFaceEncoding.objects.filter(corper_id=corper_id, session_id=session_id).count()
    if encs and current_count < MAX_ENCODINGS:
        import json
        vec = np.asarray(encs[0], dtype=np.float32)
        TempFaceEncoding.objects.create(
            corper_id=corper_id,
            session_id=session_id,
            idx=current_count,
            vector=json.dumps(vec.tolist()),
        )
        current_count += 1

    return JsonResponse({'frame': processed, 'saved': current_count})


def capture_finalize(request, corper_id: int):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    # Auth similar to capture_page
    user = request.user
    try:
        corper = CorpMember.objects.get(pk=corper_id)
    except CorpMember.DoesNotExist:
        return HttpResponseNotFound('Corper not found')

    allowed = False
    if getattr(user, 'role', None) == 'ORG' and getattr(corper.user, 'id', None) == user.id:
        allowed = True
    if getattr(user, 'role', None) == 'BRANCH':
        b = BranchOffice.objects.filter(admin=user).first()
        if b and b.id == getattr(corper.branch, 'id', None):
            allowed = True
    if getattr(user, 'role', None) == 'CORPER' and getattr(corper.account, 'id', None) == user.id:
        allowed = True
    if not allowed:
        return JsonResponse({'detail': 'Not allowed'}, status=403)

    session_id = request.POST.get('session')
    if not session_id:
        return JsonResponse({'detail': 'Missing session'}, status=400)

    # Collect encodings from DB for this session
    import json
    rows = list(TempFaceEncoding.objects.filter(corper=corper, session_id=session_id).order_by('idx', 'id'))
    if not rows:
        return JsonResponse({'detail': 'No encodings found; ensure face is visible'}, status=400)
    try:
        arrs = [np.array(json.loads(r.vector), dtype=np.float32) for r in rows]
        avg = np.mean(np.stack(arrs, axis=0), axis=0)
    except Exception:
        return JsonResponse({'detail': 'Failed to build average encoding'}, status=500)
    # Save to model as JSON list of floats
    try:
        corper.face_encoding = json.dumps(avg.tolist())
        corper.save(update_fields=['face_encoding'])
    except Exception as e:
        return JsonResponse({'detail': f'Failed to save encoding: {e}'}, status=500)
    # Delete temp rows for this session
    TempFaceEncoding.objects.filter(corper=corper, session_id=session_id).delete()

    # Reset in-memory state
    _reset_capture_state(corper_id)

    return JsonResponse({'status': 'ok', 'encodings': len(rows)})


def attendance_page(request):
    # Only corpers can mark attendance for themselves
    user = request.user
    if getattr(user, 'role', None) != 'CORPER':
        raise PermissionDenied('Only corpers can access attendance')
    cm = getattr(user, 'corper_profile', None)
    if not cm:
        return HttpResponseNotFound('Corper profile not found')

    _reset_attendance_state(cm.id)

    # Prefer request Origin if it matches configured FRONTEND_ORIGINS
    request_origin = request.headers.get('Origin')
    try:
        allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
    except Exception:
        allowed = set()
    frontend_base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', 'http://localhost:5173'))).rstrip('/')

    # Ensure CSRF cookie is set for JS POSTs from this page
    try:
        get_token(request)
    except Exception:
        pass

    ctx = {
        'corper_id': cm.id,
        'full_name': cm.full_name,
        'state_code': cm.state_code,
        'frontend_base': frontend_base,
    }
    return render(request, 'attendance.html', ctx)


def _get_target_location_for_corper(cm):
    """Return (lat, lng, source) using the strictest available center.
    Strict policy: use branch coordinates if set; otherwise fallback to organization coordinates.
    """
    try:
        br = cm.branch
        if br and br.latitude is not None and br.longitude is not None:
            return br.latitude, br.longitude, 'branch'
    except Exception:
        pass
    try:
        prof = OrganizationProfile.objects.filter(user=cm.user).first()
        if prof and prof.location_lat is not None and prof.location_lng is not None:
            return prof.location_lat, prof.location_lng, 'org'
    except Exception:
        pass
    return None, None, None


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c


def attendance_authorize(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    user = request.user
    if getattr(user, 'role', None) != 'CORPER':
        return JsonResponse({'detail': 'Not allowed'}, status=403)
    cm = getattr(user, 'corper_profile', None)
    if not cm:
        return JsonResponse({'detail': 'Corper profile not found'}, status=404)
    try:
        lat = float(request.POST.get('lat'))
        lng = float(request.POST.get('lng'))
    except Exception:
        return JsonResponse({'detail': 'Invalid coordinates'}, status=400)
    tgt_lat, tgt_lng, src = _get_target_location_for_corper(cm)
    if tgt_lat is None or tgt_lng is None:
        return JsonResponse({'allowed': False, 'detail': 'Organization location not configured'}, status=400)
    # Threshold (meters); can be tuned or moved to settings
    threshold = getattr(settings, 'ATTENDANCE_GEOFENCE_METERS', 250)
    dist = _haversine_m(lat, lng, tgt_lat, tgt_lng)
    allowed = dist <= threshold
    # Block on public holidays for the organization
    today = timezone.localdate()
    is_holiday = PublicHoliday.objects.filter(user=cm.user, start_date__lte=today, end_date__gte=today).exists()
    if is_holiday:
        return JsonResponse({'allowed': False, 'detail': 'Today is a public holiday for your organization'}, status=403)
    return JsonResponse({'allowed': allowed, 'distance_m': round(dist, 1), 'threshold_m': threshold, 'source': src or 'unknown'})


def attendance_process_frame(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    user = request.user
    if getattr(user, 'role', None) != 'CORPER':
        return JsonResponse({'ok': False, 'reason': 'not_allowed'})
    cm = getattr(user, 'corper_profile', None)
    if not cm:
        return JsonResponse({'ok': False, 'reason': 'no_profile'})
    frame = request.POST.get('frame')
    if not frame:
        return JsonResponse({'ok': False, 'reason': 'missing_frame'})

    # Load stored encoding
    import json
    try:
        saved = np.array(json.loads(cm.face_encoding), dtype='float32') if cm.face_encoding else None
    except Exception:
        saved = None
    if saved is None:
        return JsonResponse({'ok': False, 'reason': 'no_encoding'})

    # Decode incoming frame and detect face(s)
    img_data = base64.b64decode(frame)
    nparr = np.frombuffer(img_data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JsonResponse({'ok': False, 'reason': 'invalid_frame'})
    rgb = sanitize_rgb(bgr)
    _debug_img('attendance_process_frame.rgb', rgb)
    # Skip blank/dark frames early
    try:
        if rgb is None or float(np.max(rgb)) <= 5.0:
            return JsonResponse({'ok': False, 'reason': 'blank'})
    except Exception:
        return JsonResponse({'ok': False, 'reason': 'invalid_frame'})
    # Detect faces and compute encodings with locations to draw overlays
    try:
        face_locations = _safe_face_locations(rgb, tag='attendance')
        print(f"[attendance] detected faces: {len(face_locations)}")
        img_for_enc = ensure_dlib_rgb(rgb)
        _debug_img('attendance.enc_img', img_for_enc)
        encs = face_recognition.face_encodings(img_for_enc, face_locations) if img_for_enc is not None else []
    except Exception as e:
        print("[attendance] Error in face_encodings:", e)
        traceback.print_exc()
        face_locations, encs = [], []

    recognized = False
    best_conf = 0.0
    best_idx = -1
    # Iterate faces, compute distance and draw rectangles + confidence bar
    for idx, ((top, right, bottom, left), enc) in enumerate(zip(face_locations, encs)):
        dist = float(face_recognition.face_distance([saved], enc)[0])
        # Convert to 0-100 confidence; clamp
        conf = max(0.0, min(100.0, (1.0 - dist) * 100.0))
        is_match = dist <= 0.6
        # Track best
        if conf > best_conf:
            best_conf = conf
            best_idx = idx
        # Rectangle around face
        cv2.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0) if is_match else (0, 0, 255), 2)
        # Grey background bar
        cv2.rectangle(bgr, (right + 10, top), (right + 30, bottom), (128, 128, 128), -1)
        # Filled confidence portion (green if >=65 else red)
        bar_height = bottom - top
        filled = int(bar_height * (conf / 100.0))
        color = (0, 255, 0) if conf >= 65 else (0, 0, 255)
        cv2.rectangle(bgr, (right + 10, bottom - filled), (right + 30, bottom), color, -1)
        # If confident, show name + state_code box
        if conf >= 65:
            cv2.rectangle(bgr, (right + 35, top), (right + 260, top + 45), (128, 128, 128), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(bgr, f"{cm.full_name}", (right + 40, top + 40), font, 0.6, (0, 255, 0), 1)
            cv2.putText(bgr, f"{cm.state_code}", (right + 40, top + 20), font, 0.6, (0, 255, 0), 1)

    # Update recognition state using the best face confidence
    st = _ATTENDANCE_STATE.setdefault(cm.id, {'hits': 0, 'logged': False})
    if best_conf >= 65:  # ~0.35 distance equivalence; empirical
        st['hits'] = min(10, st.get('hits', 0) + 1)
        if st['hits'] >= 3:
            recognized = True
    else:
        st['hits'] = 0

    # If recognized and not yet logged, log attendance once and redirect
    if recognized and not st.get('logged'):
        # Cooldown protection using cache
        cooldown_seconds = int(getattr(settings, 'ATTENDANCE_COOLDOWN_SECONDS', 90))
        cd_key = f'attendance_cooldown:{user.id}'
        if cache.get(cd_key):
            # Still in cooldown; return silent ok:false to keep loop running client-side
            label = 'COOLDOWN'
            head_color = (0, 200, 200)
            cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_color, 2, cv2.LINE_AA)
            _, buffer = cv2.imencode('.jpg', bgr)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            return JsonResponse({'ok': False, 'reason': 'cooldown', 'frame': processed_frame})

        # Persist attendance log (same logic as finalize, without geofence)
        from .models import AttendanceLog
        now = timezone.localtime()
        today = timezone.localdate()
        # Derive state from state_code if possible
        state = ''
        try:
            state = (cm.state_code or '').split('/')[0]
        except Exception:
            state = ''
        log, created = AttendanceLog.objects.get_or_create(
            account=user,
            date=today,
            defaults={
                'org': cm.user,
                'name': cm.full_name,
                'state': state,
                'code': cm.state_code or '',
            }
        )
        if created or not log.time_in:
            log.time_in = now.time()
        if not log.time_in or now.time() >= log.time_in:
            log.time_out = now.time()
        else:
            log.time_out = log.time_in
        log.save()
        st['logged'] = True
        cache.set(cd_key, True, timeout=max(1, cooldown_seconds))

        # Return JSON success with redirect path and message
        return JsonResponse({'ok': True, 'redirect': '/dashboard/', 'message': f'Attendance logged for {cm.full_name}'})

    # Otherwise, draw overlays and return 200 JSON (no redirect)
    label = 'RECOGNIZED' if recognized else 'Scanning…'
    head_color = (0, 200, 0) if recognized else (0, 200, 200)
    cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_color, 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', bgr)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    # If no faces detected at all, hint for client
    reason = None
    if not recognized:
        if not face_locations:
            reason = 'no_face'
        else:
            reason = 'no_match'
    return JsonResponse({'ok': False, 'reason': reason or 'scanning', 'frame': processed_frame, 'recognized': recognized})


def attendance_finalize(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    user = request.user
    if getattr(user, 'role', None) != 'CORPER':
        return JsonResponse({'detail': 'Not allowed'}, status=403)
    cm = getattr(user, 'corper_profile', None)
    if not cm:
        return JsonResponse({'detail': 'Corper profile not found'}, status=404)
    st = _ATTENDANCE_STATE.get(cm.id, {'hits': 0})
    if st.get('hits', 0) < 3:
        return JsonResponse({'detail': 'Face not recognized'}, status=400)
    # Geofence check (require lat/lng in request)
    try:
        lat = float(request.POST.get('lat')) if request.method == 'POST' else None
        lng = float(request.POST.get('lng')) if request.method == 'POST' else None
    except Exception:
        lat = lng = None
    tgt_lat, tgt_lng, _ = _get_target_location_for_corper(cm)
    if tgt_lat is None or tgt_lng is None:
        return JsonResponse({'detail': 'Organization location not configured'}, status=400)
    if lat is None or lng is None:
        return JsonResponse({'detail': 'Missing location; enable location to mark attendance'}, status=400)
    threshold = getattr(settings, 'ATTENDANCE_GEOFENCE_METERS', 250)
    if _haversine_m(lat, lng, tgt_lat, tgt_lng) > threshold:
        return JsonResponse({'detail': 'You are not within the attendance proximity'}, status=403)
    # Block on public holidays
    today = timezone.localdate()
    if PublicHoliday.objects.filter(user=cm.user, start_date__lte=today, end_date__gte=today).exists():
        return JsonResponse({'detail': 'Today is a public holiday for your organization'}, status=403)
    # Persist attendance log: create/update today's record
    from .models import AttendanceLog
    now = timezone.localtime()
    # Derive state from state_code if possible
    state = ''
    try:
        state = (cm.state_code or '').split('/')[0]
    except Exception:
        state = ''
    log, created = AttendanceLog.objects.get_or_create(
        account=user,
        date=today,
        defaults={
            'org': cm.user,
            'name': cm.full_name,
            'state': state,
            'code': cm.state_code or '',
        }
    )
    # Set time_in if missing, and always refresh time_out to now (not earlier than time_in)
    if created or not log.time_in:
        log.time_in = now.time()
    # Always update time_out on check-in to reflect last seen time
    if not log.time_in or now.time() >= log.time_in:
        log.time_out = now.time()
    else:
        # Safety: never set time_out earlier than time_in
        log.time_out = log.time_in
    log.save()

    _reset_attendance_state(cm.id)
    return JsonResponse({'status': 'ok'})


class RegisterView(APIView):
    # AllowAny and disable SessionAuthentication/CSRF for this endpoint
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = OrganizationRegisterSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({
            'message': 'Registration successful. Please check your email to verify your account.'
        }, status=status.HTTP_201_CREATED)


class VerifyEmailView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        token = request.query_params.get('token')
        if not token:
            return Response({'detail': 'Missing token'}, status=status.HTTP_400_BAD_REQUEST)

        user_id = validate_email_token(token)
        if not user_id:
            return Response({'detail': 'Invalid or expired token'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

        user.is_active = True
        user.is_email_verified = True
        user.save(update_fields=['is_active', 'is_email_verified'])

        # If called from SPA (front=1), return JSON instead of redirecting
        if request.query_params.get('front') == '1':
            role = getattr(user, 'role', None)
            return Response({'verified': True, 'role': role})

        # Redirect to frontend page. If user has no usable password, include token for password setup
        # Prefer request Origin if it matches configured FRONTEND_ORIGINS for better DX (5173/5174)
        request_origin = request.headers.get('Origin')
        try:
            allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
        except Exception:
            allowed = set()
        base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', 'http://localhost:5173'))).rstrip('/')
        # If the user has no password (invited admin/corper), send to password set page with token
        role = request.query_params.get('role') or getattr(user, 'role', None)
        # For invited roles (branch admin, corper), always allow setting password on first verify
        if role in ('BRANCH', 'CORPER'):
            return redirect(f"{base}/verify-success?token={token}&role={role}")
        # For org accounts, only show password set if they don't already have one
        if not user.has_usable_password():
            return redirect(f"{base}/verify-success?token={token}")
        # Otherwise, show success page (no password needed)
        return redirect(f"{base}/verify-success")


class PasswordSetView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        token = request.data.get('token')
        password = request.data.get('password')
        password_confirm = request.data.get('password_confirm')
        if not token or not password:
            return Response({'detail': 'token and password are required'}, status=status.HTTP_400_BAD_REQUEST)
        if password_confirm is not None and password != password_confirm:
            return Response({'detail': 'Passwords do not match'}, status=status.HTTP_400_BAD_REQUEST)
        user_id = validate_email_token(token)
        if not user_id:
            return Response({'detail': 'Invalid or expired token'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        user.set_password(password)
        user.is_active = True
        user.is_email_verified = True
        user.save(update_fields=['password', 'is_active', 'is_email_verified'])
        return Response({'message': 'Password set successfully'})


class PasswordResetRequestView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'detail': 'email is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # Avoid user enumeration
            return Response({'message': 'If the email exists, a reset link has been sent.'})
        token = generate_email_token(user.id)
        base = getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')).rstrip('/')
        reset_url = f"{base}/reset-password?token={token}"
        from django.core.mail import send_mail
        send_mail(
            'Reset your NYSC Clearance password',
            f"Use the link to reset your password:\n{reset_url}",
            settings.DEFAULT_FROM_EMAIL,
            [email],
        )
        return Response({'message': 'If the email exists, a reset link has been sent.'})


class PasswordResetConfirmView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        token = request.data.get('token')
        password = request.data.get('password')
        password_confirm = request.data.get('password_confirm')
        if not token or not password:
            return Response({'detail': 'token and password are required'}, status=status.HTTP_400_BAD_REQUEST)
        if password_confirm is not None and password != password_confirm:
            return Response({'detail': 'Passwords do not match'}, status=status.HTTP_400_BAD_REQUEST)
        user_id = validate_email_token(token)
        if not user_id:
            return Response({'detail': 'Invalid or expired token'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        user.set_password(password)
        user.save(update_fields=['password'])
        return Response({'message': 'Password reset successfully'})


class CSRFView(APIView):
    permission_classes = []
    authentication_classes = []

    def get(self, request):
        # Generate CSRF token, set cookie, and return it for debugging
        token = get_token(request)
        return Response({'csrfToken': token})


class ConfigView(APIView):
    """Expose non-sensitive runtime config for the frontend.

    Helps centralize CSRF/cookie names, base URLs, and origins across environments.
    """
    permission_classes = []
    authentication_classes = []

    def get(self, request):
        try:
            cors = list(getattr(settings, 'CORS_ALLOWED_ORIGINS', []) or [])
        except Exception:
            cors = []
        try:
            csrf_origins = list(getattr(settings, 'CSRF_TRUSTED_ORIGINS', []) or [])
        except Exception:
            csrf_origins = []
        data = {
            'api_base': getattr(settings, 'API_BASE_URL', ''),
            'frontend_base': getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', '')),
            'csrf_cookie_name': getattr(settings, 'CSRF_COOKIE_NAME', 'csrftoken'),
            'session_cookie_name': getattr(settings, 'SESSION_COOKIE_NAME', 'sessionid'),
            'cors_allowed_origins': cors,
            'csrf_trusted_origins': csrf_origins,
            'cookie_same_site': {
                'session': getattr(settings, 'SESSION_COOKIE_SAMESITE', 'Lax'),
                'csrf': getattr(settings, 'CSRF_COOKIE_SAMESITE', 'Lax'),
            },
            'cookie_secure': {
                'session': bool(getattr(settings, 'SESSION_COOKIE_SECURE', False)),
                'csrf': bool(getattr(settings, 'CSRF_COOKIE_SECURE', False)),
            },
            'debug': bool(getattr(settings, 'DEBUG', False)),
        }
        return Response(data)


class LoginView(APIView):
    # AllowAny and disable SessionAuthentication/CSRF for this endpoint
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        email = serializer.validated_data['email']
        password = serializer.validated_data['password']
        expected_role = serializer.validated_data.get('role')
        user = authenticate(request, email=email, password=password)
        if not user:
            return Response({'detail': 'Invalid credentials'}, status=status.HTTP_400_BAD_REQUEST)
        if expected_role and getattr(user, 'role', None) != expected_role:
            return Response({'detail': 'Invalid role for this login'}, status=status.HTTP_403_FORBIDDEN)
        if not user.is_active:
            return Response({'detail': 'Account not active'}, status=status.HTTP_403_FORBIDDEN)
        # Issue stateless token while keeping session login for backward compatibility
        try:
            from .auth import make_access_token
            token = make_access_token(user.id)
        except Exception:
            token = None
        # Optional: session login (kept for compatibility with existing clients)
        try:
            login(request, user)
        except Exception:
            pass
        OrganizationProfile.objects.get_or_create(user=user)
        resp = {'message': 'Logged in'}
        if token:
            resp['token'] = token
            resp['token_type'] = 'Bearer'
            resp['expires_in'] = 60 * 60 * 24
        return Response(resp)


class LogoutView(APIView):
    # Accept both session and token-based clients; no CSRF enforcement
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        logout(request)
        return Response({'message': 'Logged out'})


class MeView(APIView):
    def get(self, request):
        if not request.user.is_authenticated:
            return Response({'authenticated': False})
        return Response({
            'authenticated': True,
            'email': request.user.email,
            'name': request.user.name,
            'role': getattr(request.user, 'role', 'ORG'),
        })


class ProfileView(APIView):
    def get(self, request):
        user = request.user
        org_user = user
        if getattr(user, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            if b:
                org_user = b.user
        elif getattr(user, 'role', None) == 'CORPER':
            try:
                org_user = user.corper_profile.user
            except Exception:
                pass
        profile, _ = OrganizationProfile.objects.get_or_create(user=org_user)
        return Response(OrganizationProfileSerializer(profile).data)

    def put(self, request):
        if getattr(request.user, 'role', None) != 'ORG':
            raise PermissionDenied('Only organization can update profile')
        profile, _ = OrganizationProfile.objects.get_or_create(user=request.user)
        serializer = OrganizationProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class BranchOfficeViewSet(viewsets.ModelViewSet):
    serializer_class = BranchOfficeSerializer

    def get_queryset(self):
        user = self.request.user
        if getattr(user, 'role', None) == 'ORG':
            return BranchOffice.objects.filter(user=user)
        if getattr(user, 'role', None) == 'BRANCH':
            return BranchOffice.objects.filter(admin=user)
        if getattr(user, 'role', None) == 'CORPER':
            try:
                br_id = user.corper_profile.branch_id
                return BranchOffice.objects.filter(id=br_id)
            except Exception:
                return BranchOffice.objects.none()
        return BranchOffice.objects.none()

    def perform_create(self, serializer):
        # user is set inside the serializer using request from context
        serializer.save()

    @action(detail=True, methods=['post'])
    def clone_structure(self, request, pk=None):
        """Copy departments and units from a source branch into this branch.

        Body: { "source": <branch_id> }
        - If a department with the same name exists in the target, reuse it.
        - If a unit with the same name exists under the matched department, reuse it.
        """
        try:
            target = BranchOffice.objects.get(pk=pk)
        except BranchOffice.DoesNotExist:
            return Response({ 'detail': 'Target branch not found' }, status=status.HTTP_404_NOT_FOUND)

        source_id = request.data.get('source') or request.data.get('source_branch')
        if not source_id:
            return Response({ 'detail': 'source is required' }, status=status.HTTP_400_BAD_REQUEST)
        try:
            source = BranchOffice.objects.get(pk=source_id)
        except BranchOffice.DoesNotExist:
            return Response({ 'detail': 'Source branch not found' }, status=status.HTTP_404_NOT_FOUND)

        # Authorization: ensure both branches belong to the same organization for the requester
        user = request.user
        if getattr(user, 'role', None) == 'ORG':
            if target.user_id != user.id or source.user_id != user.id:
                raise PermissionDenied('Branches must belong to your organization')
        elif getattr(user, 'role', None) == 'BRANCH':
            # Branch admin can clone into their own branch only, and only from within same org
            if target.admin_id != user.id:
                raise PermissionDenied('Not allowed for this target branch')
            if source.user_id != target.user_id:
                raise PermissionDenied('Source must belong to the same organization')
        else:
            raise PermissionDenied('Not allowed')

        from .models import Department, Unit
        created_deps = 0
        created_units = 0
        # Map department name -> department instance for target
        existing_target_deps = { d.name: d for d in Department.objects.filter(branch=target) }
        for d in Department.objects.filter(branch=source).order_by('name'):
            t_dep = existing_target_deps.get(d.name)
            if not t_dep:
                t_dep = Department.objects.create(branch=target, name=d.name)
                existing_target_deps[d.name] = t_dep
                created_deps += 1
            # Units under department
            existing_units = { u.name: u for u in Unit.objects.filter(department=t_dep) }
            for u in Unit.objects.filter(department=d).order_by('name'):
                if u.name in existing_units:
                    continue
                Unit.objects.create(department=t_dep, name=u.name)
                created_units += 1

        return Response({ 'created_departments': created_deps, 'created_units': created_units })


class DepartmentViewSet(viewsets.ModelViewSet):
    serializer_class = DepartmentSerializer
    def get_queryset(self):
        user = self.request.user
        if getattr(user, 'role', None) == 'ORG':
            return Department.objects.filter(branch__user=user)
        if getattr(user, 'role', None) == 'BRANCH':
            return Department.objects.filter(branch__admin=user)
        if getattr(user, 'role', None) == 'CORPER':
            try:
                br = user.corper_profile.branch
                return Department.objects.filter(branch=br)
            except Exception:
                return Department.objects.none()
        return Department.objects.none()

    def perform_create(self, serializer):
        # ensure branch belongs to org or is managed by branch admin
        branch = serializer.validated_data.get('branch')
        user = self.request.user
        if getattr(user, 'role', None) == 'ORG':
            if branch.user_id != user.id:
                raise PermissionDenied('Invalid branch')
        elif getattr(user, 'role', None) == 'BRANCH':
            if branch.admin_id != user.id:
                raise PermissionDenied('Invalid branch for this admin')
        else:
            raise PermissionDenied('Not allowed')
        serializer.save()


class UnitViewSet(viewsets.ModelViewSet):
    serializer_class = UnitSerializer
    def get_queryset(self):
        user = self.request.user
        if getattr(user, 'role', None) == 'ORG':
            return Unit.objects.filter(department__branch__user=user)
        if getattr(user, 'role', None) == 'BRANCH':
            return Unit.objects.filter(department__branch__admin=user)
        if getattr(user, 'role', None) == 'CORPER':
            try:
                br = user.corper_profile.branch
                return Unit.objects.filter(department__branch=br)
            except Exception:
                return Unit.objects.none()
        return Unit.objects.none()

    def perform_create(self, serializer):
        dept = serializer.validated_data.get('department')
        user = self.request.user
        if getattr(user, 'role', None) == 'ORG':
            if dept.branch.user_id != user.id:
                raise PermissionDenied('Invalid department')
        elif getattr(user, 'role', None) == 'BRANCH':
            if dept.branch.admin_id != user.id:
                raise PermissionDenied('Invalid department for this admin')
        else:
            raise PermissionDenied('Not allowed')
        serializer.save()


class CorpMemberViewSet(viewsets.ModelViewSet):
    serializer_class = CorpMemberSerializer

    def get_queryset(self):
        user = self.request.user
        if user.role == 'ORG':
            # Include any corpers owned by this org or whose branch belongs to this org
            return CorpMember.objects.filter(models.Q(user=user) | models.Q(branch__user=user)).distinct()
        if user.role == 'BRANCH':
            branches = BranchOffice.objects.filter(admin=user)
            return CorpMember.objects.filter(branch__in=branches)
        if user.role == 'CORPER':
            try:
                return CorpMember.objects.filter(account=user)
            except Exception:
                return CorpMember.objects.none()
        return CorpMember.objects.none()

    def perform_create(self, serializer):
        # Serializer will map the owning organization correctly and default branch for branch admins
        serializer.save()

    def perform_update(self, serializer):
        """Restrict branch admins to editing only their corpers and to their own branches/departments/units."""
        user = self.request.user
        instance = self.get_object()
        if getattr(user, 'role', None) == 'BRANCH':
            # Ensure the corper is in one of the admin's branches
            admin_branch = BranchOffice.objects.filter(admin=user)
            if not admin_branch.filter(id=getattr(instance.branch, 'id', None)).exists():
                raise PermissionDenied('Not allowed to edit this corper')
            # If updating branch/department/unit, constrain them to admin's branches
            data = serializer.validated_data
            new_branch = data.get('branch') or instance.branch
            if new_branch and not admin_branch.filter(id=new_branch.id).exists():
                raise PermissionDenied('Cannot move corper outside your branch')
            # Validate department/unit consistency if provided
            dept = data.get('department')
            unit = data.get('unit')
            if dept and new_branch and getattr(dept, 'branch_id', None) != getattr(new_branch, 'id', None):
                raise PermissionDenied('Department does not belong to your branch')
            if unit and dept and getattr(unit, 'department_id', None) != getattr(dept, 'id', None):
                raise PermissionDenied('Unit does not belong to the selected department')
        serializer.save()


class PublicHolidayViewSet(viewsets.ModelViewSet):
    serializer_class = PublicHolidaySerializer

    def get_queryset(self):
        user = self.request.user
        if getattr(user, 'role', 'ORG') == 'ORG':
            return PublicHoliday.objects.filter(user=user).order_by('start_date')
        # For branch admin/corper, map to their org
        org_user = None
        if user.role == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            org_user = b.user if b else None
        elif user.role == 'CORPER':
            try:
                org_user = user.corper_profile.user
            except Exception:
                org_user = None
        return PublicHoliday.objects.filter(user=org_user).order_by('start_date')

    def perform_create(self, serializer):
        if self.request.user.role != 'ORG':
            raise PermissionDenied('Only organization can create holidays')
        serializer.save(user=self.request.user)


class LeaveRequestViewSet(viewsets.ModelViewSet):
    serializer_class = LeaveRequestSerializer

    def get_queryset(self):
        user = self.request.user
        if user.role == 'ORG':
            return LeaveRequest.objects.filter(corper__user=user).order_by('-created_at')
        if user.role == 'BRANCH':
            branches = BranchOffice.objects.filter(admin=user)
            return LeaveRequest.objects.filter(branch__in=branches).order_by('-created_at')
        if user.role == 'CORPER':
            try:
                cm = user.corper_profile
            except Exception:
                return LeaveRequest.objects.none()
            return LeaveRequest.objects.filter(corper=cm).order_by('-created_at')
        return LeaveRequest.objects.none()

    def perform_create(self, serializer):
        user = self.request.user
        if user.role != 'CORPER':
            raise PermissionDenied('Only corpers can create leave requests')
        cm = getattr(user, 'corper_profile', None)
        if not cm:
            raise PermissionDenied('Corper profile not found')
        serializer.save(corper=cm, branch=cm.branch)

    @action(detail=True, methods=['post'])
    def approve(self, request, pk=None):
        lr = self.get_object()
        user = request.user
        if user.role not in ('ORG','BRANCH'):
            raise PermissionDenied('Not allowed')
        if user.role == 'BRANCH' and lr.branch and lr.branch.admin_id != user.id:
            raise PermissionDenied('Not your branch')
        lr.status = 'APPROVED'
        lr.decided_by = user
        lr.save(update_fields=['status','decided_by','updated_at'])
        return Response({'status': 'APPROVED'})

    @action(detail=True, methods=['post'])
    def reject(self, request, pk=None):
        lr = self.get_object()
        user = request.user
        if user.role not in ('ORG','BRANCH'):
            raise PermissionDenied('Not allowed')
        if user.role == 'BRANCH' and lr.branch and lr.branch.admin_id != user.id:
            raise PermissionDenied('Not your branch')
        lr.status = 'REJECTED'
        lr.decided_by = user
        lr.save(update_fields=['status','decided_by','updated_at'])
        return Response({'status': 'REJECTED'})


class StatsView(APIView):
    def get(self, request):
        user = request.user
        # Scope stats by role: ORG sees org-wide, BRANCH sees their managed branch only
        if getattr(user, 'role', None) == 'BRANCH':
            branch_qs = BranchOffice.objects.filter(admin=user)
            corpers_qs = CorpMember.objects.filter(branch__in=branch_qs)
            departments_qs = Department.objects.filter(branch__in=branch_qs)
            units_qs = Unit.objects.filter(department__branch__in=branch_qs)
            # Attendance logs for accounts under these branches
            att_qs = AttendanceLog.objects.filter(account__corper_profile__branch__in=branch_qs)
        else:
            branch_qs = BranchOffice.objects.filter(user=user)
            corpers_qs = CorpMember.objects.filter(user=user)
            departments_qs = Department.objects.filter(branch__user=user)
            units_qs = Unit.objects.filter(department__branch__user=user)
            # Attendance logs for this organization
            if getattr(user, 'role', None) == 'CORPER':
                att_qs = AttendanceLog.objects.filter(account=user)
            else:
                att_qs = AttendanceLog.objects.filter(org=user)

        by_branch_qs = (
            corpers_qs.values('branch__name')
                      .annotate(count=Count('id'))
                      .order_by('branch__name')
        )
        corpers_by_branch = []
        for row in by_branch_qs:
            name = row['branch__name'] or 'Unassigned'
            corpers_by_branch.append({'branch': name, 'count': row['count']})

        data = {
            'totals': {
                'branches': branch_qs.count(),
                'departments': departments_qs.count(),
                'units': units_qs.count(),
                'corpers': corpers_qs.count(),
            },
            'corpers_by_branch': corpers_by_branch,
            'attendance': _attendance_stats(att_qs)
        }
        return Response(data)

def _attendance_stats(att_qs):
    """Return counts for today, this month, and last 7 days timeline.

    Also includes per-day hours (float) computed where both time_in and time_out are present.
    For org/branch scopes (multiple accounts), hours are summed per date.
    """
    today = timezone.localdate()
    start_month = today.replace(day=1)
    today_qs = att_qs.filter(date=today)
    today_count = today_qs.count()
    month_qs = att_qs.filter(date__gte=start_month, date__lte=today)
    month_count = month_qs.count()
    last7 = []
    for i in range(6, -1, -1):
        d = today - timezone.timedelta(days=i)
        day_qs = att_qs.filter(date=d)
        c = day_qs.count()
        # Sum hours across logs for the date
        hours = 0.0
        for log in day_qs:
            if log.time_in and log.time_out:
                from datetime import datetime
                try:
                    start_dt = datetime.combine(d, log.time_in)
                    end_dt = datetime.combine(d, log.time_out)
                    delta = end_dt - start_dt
                    if delta.total_seconds() > 0:
                        hours += round(delta.total_seconds() / 3600.0, 2)
                except Exception:
                    pass
        last7.append({'date': d.isoformat(), 'count': c, 'hours': round(hours, 2)})
    return {
        'today': today_count,
        'this_month': month_count,
        'last7': last7,
    }


def _prev_month_bounds(today=None):
    today = today or timezone.localdate()
    first_this = today.replace(day=1)
    last_prev = first_this - timezone.timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev


def _working_days(user, start_date, end_date):
    """Return list of working dates excluding weekends and org public holidays."""
    days = []
    d = start_date
    holidays = PublicHoliday.objects.filter(user=user, start_date__lte=end_date, end_date__gte=start_date)
    def is_holiday(x):
        return holidays.filter(start_date__lte=x, end_date__gte=x).exists()
    while d <= end_date:
        if d.weekday() < 5 and not is_holiday(d):
            days.append(d)
        d += timezone.timedelta(days=1)
    return days


def verify_clearance(request):
    """Public endpoint to verify a clearance reference.

    Expected format: NYSC-<STATE_CODE>-<YYYYMM>
    Example: NYSC-FC/23C/2354-202510
    """
    ref = request.GET.get('ref', '').strip()
    ctx = { 'ref': ref, 'valid': False, 'error': None }
    try:
        if not ref.startswith('NYSC-'):
            ctx['error'] = 'Invalid reference format'
            return render(request, 'verify_clearance.html', ctx, status=400)
        rest = ref[5:]
        dash = rest.rfind('-')
        if dash == -1:
            ctx['error'] = 'Invalid reference format'
            return render(request, 'verify_clearance.html', ctx, status=400)
        state_code = rest[:dash]
        yyyymm = rest[dash+1:]
        if len(yyyymm) != 6 or not yyyymm.isdigit():
            ctx['error'] = 'Invalid month segment'
            return render(request, 'verify_clearance.html', ctx, status=400)
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])
        if month < 1 or month > 12:
            ctx['error'] = 'Invalid month value'
            return render(request, 'verify_clearance.html', ctx, status=400)

        # Locate the corper by state code
        cm = CorpMember.objects.filter(state_code=state_code).select_related('user', 'account', 'branch').first()
        if not cm:
            ctx['error'] = 'Reference not found'
            return render(request, 'verify_clearance.html', ctx, status=404)

        # Compute the month bounds for provided year-month
        from datetime import date
        import calendar
        start = date(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end = date(year, month, last_day)

        # Basic corroboration: ensure attendance logs exist in that month for the account (optional)
        has_any_log = AttendanceLog.objects.filter(account=cm.account, date__gte=start, date__lte=end).exists()

        # Organization details / signatory
        prof = OrganizationProfile.objects.filter(user=cm.user).first()
        signatory_name = getattr(prof, 'signatory_name', '') if prof else ''
        signature_url = getattr(prof.signature, 'url', '') if prof and getattr(prof, 'signature', None) else ''

        ctx.update({
            'valid': True,
            'state_code': cm.state_code,
            'name': cm.full_name,
            'org_name': getattr(cm.user, 'name', ''),
            'year': year,
            'month': start.strftime('%B'),
            'has_any_log': has_any_log,
            'signatory_name': signatory_name,
            'signature_url': signature_url,
        })
        return render(request, 'verify_clearance.html', ctx)
    except Exception:
        ctx['error'] = 'An unexpected error occurred'
        return render(request, 'verify_clearance.html', ctx, status=500)


def performance_summary(request):
    if getattr(request.user, 'role', None) != 'CORPER':
        raise PermissionDenied('Only corpers can access')
    cm = getattr(request.user, 'corper_profile', None)
    if not cm:
        return JsonResponse({'detail': 'Corper profile not found'}, status=404)
    start, end = _prev_month_bounds()
    work_days = _working_days(cm.user, start, end)
    work_set = set(work_days)
    logs = AttendanceLog.objects.filter(account=request.user, date__gte=start, date__lte=end)

    present_dates = set(logs.values_list('date', flat=True))
    present = len(present_dates & work_set)

    # Late threshold: org profile late_time; if missing, consider none late
    prof = OrganizationProfile.objects.filter(user=cm.user).first()
    late_time = getattr(prof, 'late_time', None)
    late = 0
    if late_time:
        for log in logs:
            if log.date in work_set and log.time_in and log.time_in > late_time:
                late += 1
    absent = max(0, len(work_days) - present)
    on_time = max(0, present - late)

    data = {
        'month': start.strftime('%B %Y').upper(),
        'range': {'start': start.isoformat(), 'end': end.isoformat()},
        'working_days': len(work_days),
        'present': present,
        'absent': absent,
        'late': late,
        'on_time': on_time,
        'name': cm.full_name.upper(),
        'state_code': cm.state_code,
    }
    return JsonResponse(data)


def performance_clearance_page(request):
    if getattr(request.user, 'role', None) != 'CORPER':
        raise PermissionDenied('Only corpers can access')
    cm = getattr(request.user, 'corper_profile', None)
    if not cm:
        return HttpResponseNotFound('Corper profile not found')
    start, end = _prev_month_bounds()
    # Organization logo
    prof = OrganizationProfile.objects.filter(user=cm.user).first()
    logo_url = getattr(prof.logo, 'url', '') if prof and getattr(prof, 'logo', None) else ''
    # Signatory details from organization profile
    signature_url = getattr(prof.signature, 'url', '') if prof and getattr(prof, 'signature', None) else ''

    # Pronoun based on gender (possessive)
    gender = (cm.gender or '').upper()
    if gender == 'M':
        pronoun = 'his'
    elif gender == 'F':
        pronoun = 'her'
    else:
        pronoun = 'their'

    # Organization contact details
    org_address = getattr(cm.user, 'address', '') or ''
    org_email = getattr(cm.user, 'email', '') or ''
    org_phone = getattr(cm.user, 'phone_number', '') or ''

    ref_number = f"NYSC-{cm.state_code}-{start.strftime('%Y%m')}"
    verification_url = request.build_absolute_uri(f'/verify/?ref={ref_number}')

    # Eligibility check: previous month lateness/absence vs org thresholds
    # Skip this check if it's the corper's first clearance (no prior clearance debits found)
    is_first_clearance = not WalletTransaction.objects.filter(
        type='DEBIT', reference__startswith=f"NYSC-{cm.state_code}-"
    ).exists()
    # Allow if override exists for this month
    yyyymm = start.strftime('%Y%m')
    has_override = ClearanceOverride.objects.filter(corper=cm, year_month=yyyymm).exists()
    if not is_first_clearance and not has_override:
        work_days = _working_days(cm.user, start, end)
        work_set = set(work_days)
        logs = AttendanceLog.objects.filter(account=request.user, date__gte=start, date__lte=end)
        present_dates = set(logs.values_list('date', flat=True))
        present = len(present_dates & work_set)
        late_time = getattr(prof, 'late_time', None)
        late = 0
        if late_time:
            for log in logs:
                if log.date in work_set and log.time_in and log.time_in > late_time:
                    late += 1
        absent = max(0, len(work_days) - present)
        exceeded_absent = (
            getattr(prof, 'max_days_absent', None) is not None and absent > (prof.max_days_absent or 0)
        )
        exceeded_late = (
            getattr(prof, 'max_days_late', None) is not None and late > (prof.max_days_late or 0)
        )
        if exceeded_absent or exceeded_late:
            # Compute frontend base for links
            request_origin = request.headers.get('Origin')
            try:
                allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
            except Exception:
                allowed = set()
            frontend_base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', 'http://localhost:5173'))).rstrip('/')

            return render(request, 'clearance_restriction.html', {
                'month': start.strftime('%B %Y'),
                'absent': absent,
                'late': late,
                'max_absent': getattr(prof, 'max_days_absent', None),
                'max_late': getattr(prof, 'max_days_late', None),
                'contact_url': f'{frontend_base}/dashboard',
            }, status=403)

    # Ensure wallet and charge once per corper per month on first view
    charged = False
    try:
        _ensure_wallet_with_welcome(cm.user)
        def _charge_clearance_if_needed(org_user, reference):
            # Avoid double-charging: check any of the wallets already charged with this reference
            exists = WalletTransaction.objects.filter(reference=reference[:64], type='DEBIT').exists()
            if not exists:
                # Base amount from System Settings (fallback to constant)
                try:
                    from .models import SystemSetting
                    settings = SystemSetting.current()
                    amount = settings.clearance_fee or CLEARANCE_FEE
                except Exception:
                    amount = CLEARANCE_FEE
                # Apply discount if enabled in SystemSetting
                try:
                    if getattr(settings, 'discount_enabled', False):
                        pct = Decimal(str(settings.discount_percent or '0'))
                        if pct > 0:
                            amount = (amount * (Decimal('100') - pct) / Decimal('100')).quantize(Decimal('0.01'))
                except Exception:
                    pass
                vat = (amount * VAT_RATE).quantize(Decimal('0.01'))
                total = amount + vat
                # Attempt in order: org -> branch -> corper
                branch_user = getattr(cm.branch, 'admin', None)
                corper_user = request.user

                def try_debit(user, desc):
                    if not user:
                        return False
                    acct = _ensure_wallet_with_welcome(user)
                    if (acct.balance or Decimal('0.00')) >= total:
                        WalletTransaction.objects.create(
                            account=acct,
                            type='DEBIT',
                            amount=amount,
                            vat_amount=vat,
                            total_amount=total,
                            description=desc,
                            reference=reference[:64]
                        )
                        acct.balance = acct.balance - total
                        acct.save(update_fields=['balance'])
                        return True
                    return False

                return (
                    try_debit(org_user, 'Clearance view charge (org)') or
                    try_debit(branch_user, 'Clearance view charge (branch)') or
                    try_debit(corper_user, 'Clearance view charge (corper)')
                )
            return True
        charged = _charge_clearance_if_needed(cm.user, ref_number)
    except Exception:
        charged = False

    if not charged:
        # Deny access nicely with a prompt to fund wallet
        # Compute frontend base for links
        request_origin = request.headers.get('Origin')
        try:
            allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
        except Exception:
            allowed = set()
        frontend_base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', getattr(settings, 'FRONTEND_URL', 'http://localhost:5173'))).rstrip('/')

        return render(request, 'clearance_payment_required.html', {
            'reason': 'Insufficient wallet balance across organization, branch, and personal wallets.',
            'fund_url': f'{frontend_base}/dashboard?fund=1',
            'dashboard_url': f'{frontend_base}/dashboard',
        }, status=402)

    # Map NYSC two-letter state code to state/capital for template placeholders
    STATE_MAP = {
        'AB': {'state': 'Abia', 'capital': 'Umuahia'},
        'AD': {'state': 'Adamawa', 'capital': 'Yola'},
        'AK': {'state': 'Akwa Ibom', 'capital': 'Uyo'},
        'AN': {'state': 'Anambra', 'capital': 'Awka'},
        'BA': {'state': 'Bauchi', 'capital': 'Bauchi'},
        'BY': {'state': 'Bayelsa', 'capital': 'Yenagoa'},
        'BN': {'state': 'Benue', 'capital': 'Makurdi'},
        'BO': {'state': 'Borno', 'capital': 'Maiduguri'},
        'CR': {'state': 'Cross River', 'capital': 'Calabar'},
        'DT': {'state': 'Delta', 'capital': 'Asaba'},
        'EB': {'state': 'Ebonyi', 'capital': 'Abakaliki'},
        'ED': {'state': 'Edo', 'capital': 'Benin City'},
        'EK': {'state': 'Ekiti', 'capital': 'Ado-Ekiti'},
        'EN': {'state': 'Enugu', 'capital': 'Enugu'},
        'FC': {'state': 'FCT', 'capital': 'Abuja'},
        'GM': {'state': 'Gombe', 'capital': 'Gombe'},
        'IM': {'state': 'Imo', 'capital': 'Owerri'},
        'JG': {'state': 'Jigawa', 'capital': 'Dutse'},
        'KD': {'state': 'Kaduna', 'capital': 'Kaduna'},
        'KN': {'state': 'Kano', 'capital': 'Kano'},
        'KT': {'state': 'Katsina', 'capital': 'Katsina'},
        'KB': {'state': 'Kebbi', 'capital': 'Birnin Kebbi'},
        'KG': {'state': 'Kogi', 'capital': 'Lokoja'},
        'KW': {'state': 'Kwara', 'capital': 'Ilorin'},
        'LA': {'state': 'Lagos', 'capital': 'Ikeja'},
        'NS': {'state': 'Nasarawa', 'capital': 'Lafia'},
        'NG': {'state': 'Niger', 'capital': 'Minna'},
        'OG': {'state': 'Ogun', 'capital': 'Abeokuta'},
        'OD': {'state': 'Ondo', 'capital': 'Akure'},
        'OS': {'state': 'Osun', 'capital': 'Osogbo'},
        'OY': {'state': 'Oyo', 'capital': 'Ibadan'},
        'PL': {'state': 'Plateau', 'capital': 'Jos'},
        'RV': {'state': 'Rivers', 'capital': 'Port Harcourt'},
        'SO': {'state': 'Sokoto', 'capital': 'Sokoto'},
        'TR': {'state': 'Taraba', 'capital': 'Jalingo'},
        'YB': {'state': 'Yobe', 'capital': 'Damaturu'},
        'ZM': {'state': 'Zamfara', 'capital': 'Gusau'},
    }

    raw_code = (cm.state_code or '').upper()
    # Extract first two letters as state code (robust to formats like LA/20B/1234)
    code_letters = ''.join([ch for ch in raw_code if ch.isalpha()])[:2]
    region = STATE_MAP.get(code_letters, {'state': '', 'capital': ''})

    ctx = {
        'reference_number': ref_number,
        'date': timezone.localdate().strftime('%Y-%m-%d'),
        'month': start.strftime('%B').upper(),
        'year': start.strftime('%Y'),
        'name': cm.full_name.upper(),
        'state_code': cm.state_code,
        'state': region['state'],
        'capital': region['capital'],
        'signatory_name': (getattr(prof, 'signatory_name', '') if prof else ''),
        'pronoun': pronoun,
        'logo_url': logo_url,
        'signature_url': signature_url,
        'verification_url': verification_url,
        'org_address': org_address,
        'org_email': org_email,
        'org_phone': org_phone,
    }
    return render(request, 'performance_clearance.html', ctx)

    def _attendance_stats(self, att_qs):
        """Return counts for today, this month, and last 7 days timeline."""
        today = timezone.localdate()
        start_month = today.replace(day=1)
        # Today count: any log present (time_in set counts as present)
        today_count = att_qs.filter(date=today).count()
        month_count = att_qs.filter(date__gte=start_month, date__lte=today).count()
        # Last 7 days (inclusive today)
        last7 = []
        for i in range(6, -1, -1):
            d = today - timezone.timedelta(days=i)
            c = att_qs.filter(date=d).count()
            last7.append({'date': d.isoformat(), 'count': c})
        return {
            'today': today_count,
            'this_month': month_count,
            'last7': last7,
        }


class NotificationViewSet(viewsets.ModelViewSet):
    serializer_class = NotificationSerializer

    def get_queryset(self):
        user = self.request.user
        if user.role == 'ORG':
            return Notification.objects.filter(user=user).order_by('-created_at')
        if user.role == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            org = b.user if b else None
            return Notification.objects.filter(user=org).order_by('-created_at')
        if user.role == 'CORPER':
            try:
                cm = user.corper_profile
            except Exception:
                return Notification.objects.none()
            return Notification.objects.filter(user=cm.user).filter(models.Q(branch__isnull=True) | models.Q(branch=cm.branch)).order_by('-created_at')
        return Notification.objects.none()

    def perform_create(self, serializer):
        user = self.request.user
        if user.role == 'ORG':
            serializer.save(user=user, created_by=user)
            return
        if user.role == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            if not b:
                raise PermissionDenied('No managed branch')
            serializer.save(user=b.user, branch=b, created_by=user)
            return
        raise PermissionDenied('Not allowed')


from decimal import Decimal

VAT_RATE = Decimal('0.075')  # 7.5%
CLEARANCE_FEE = Decimal('300.00')


def _ensure_wallet_with_welcome(user):
    acct, created = WalletAccount.objects.get_or_create(user=user)
    # Apply welcome bonus only for organization users
    if getattr(user, 'role', None) == 'ORG' and (created or not acct.transactions.exists()):
        from .models import SystemSetting
        settings = SystemSetting.current()
        welcome = settings.welcome_bonus or Decimal('0.00')
        if welcome > 0:
            vat = Decimal('0.00')
            total = welcome
            WalletTransaction.objects.create(
                account=acct,
                type='CREDIT',
                amount=welcome,
                vat_amount=vat,
                total_amount=total,
                description='Welcome bonus',
                reference='WELCOME'
            )
            acct.balance = (acct.balance or Decimal('0.00')) + total
            acct.save(update_fields=['balance'])
    return acct


class WalletView(APIView):
    def get(self, request):
        user = request.user
        # Allow ORG, BRANCH, CORPER to view their own wallet
        if getattr(user, 'role', None) not in ('ORG', 'BRANCH', 'CORPER'):
            raise PermissionDenied('Not allowed')
        acct = _ensure_wallet_with_welcome(user)
        from .serializers import WalletAccountSerializer
        data = WalletAccountSerializer(acct).data
        return Response(data)


def wallet_charge_clearance(request):
    """Charge organization when a corper downloads clearance letter.

    Expects POST with JSON: { "reference": "NYSC-..." }
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    user = request.user
    if getattr(user, 'role', None) != 'CORPER':
        return JsonResponse({'detail': 'Only corper can trigger charge'}, status=403)
    try:
        cm = user.corper_profile
    except Exception:
        return JsonResponse({'detail': 'Corper profile not found'}, status=404)
    import json
    try:
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        payload = {}
    ref = payload.get('reference', '')

    # Compute charge amount
    from .models import SystemSetting
    settings = SystemSetting.current()
    try:
        amount = settings.clearance_fee or CLEARANCE_FEE
    except Exception:
        amount = CLEARANCE_FEE
    if getattr(settings, 'discount_enabled', False):
        try:
            pct = Decimal(str(settings.discount_percent or '0'))
            if pct > 0:
                amount = (amount * (Decimal('100') - pct) / Decimal('100')).quantize(Decimal('0.01'))
        except Exception:
            pass
    vat = (amount * VAT_RATE).quantize(Decimal('0.01'))
    total = amount + vat

    # Try charge from ORG -> BRANCH -> CORPER wallet
    org_user = cm.user
    branch_user = getattr(cm.branch, 'admin', None)
    corper_user = request.user

    def try_debit(user, description):
        if not user:
            return False
        acct = _ensure_wallet_with_welcome(user)
        if (acct.balance or Decimal('0.00')) >= total:
            WalletTransaction.objects.create(
                account=acct,
                type='DEBIT',
                amount=amount,
                vat_amount=vat,
                total_amount=total,
                description=description,
                reference=ref[:64]
            )
            acct.balance = acct.balance - total
            acct.save(update_fields=['balance'])
            return True
        return False

    if try_debit(org_user, 'Clearance download charge (org)') or \
       try_debit(branch_user, 'Clearance download charge (branch)') or \
       try_debit(corper_user, 'Clearance download charge (corper)'):
        return JsonResponse({'status': 'charged'})
    return JsonResponse({'status': 'insufficient', 'detail': 'Insufficient funds. Please fund your wallet or contact branch admin.'}, status=402)


class AnnouncementView(APIView):
    def get(self, request):
        user = request.user
        # Only organization dashboard shows this floating announcement
        if getattr(user, 'role', None) != 'ORG':
            return Response(status=204)
        from .models import SystemSetting
        from django.utils import timezone
        s = SystemSetting.current()
        if not getattr(s, 'notify_enabled', False):
            return Response(status=204)
        now = timezone.now()
        if s.notify_start and now < s.notify_start:
            return Response(status=204)
        if s.notify_end and now > s.notify_end:
            return Response(status=204)
        data = {
            'title': s.notify_title or 'Notice',
            'message': s.notify_message or '',
        }
        return Response(data)


class ClearanceStatusView(APIView):
    """List corpers with clearance qualification and download status for previous month.

    - ORG: all corpers under organization
    - BRANCH: corpers under managed branch
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        role = getattr(user, 'role', None)
        if role not in ('ORG', 'BRANCH'):
            raise PermissionDenied('Only organization or branch admins can view clearance status')
        if role == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            if not b:
                return Response([], status=200)
            corpers = CorpMember.objects.filter(branch=b).select_related('user', 'branch')
            org_user = b.user
        else:
            corpers = CorpMember.objects.filter(user=user).select_related('user', 'branch')
            org_user = user
        start, end = _prev_month_bounds()
        yyyymm = start.strftime('%Y%m')
        prof = OrganizationProfile.objects.filter(user=org_user).first()
        late_time = getattr(prof, 'late_time', None)
        max_absent = getattr(prof, 'max_days_absent', None)
        max_late = getattr(prof, 'max_days_late', None)

        # Preload attendance logs for the window
        acc_ids = list(corpers.values_list('account_id', flat=True))
        logs = AttendanceLog.objects.filter(account_id__in=acc_ids, date__gte=start, date__lte=end)
        logs_by_acc = {}
        for lg in logs:
            logs_by_acc.setdefault(lg.account_id, []).append(lg)

        results = []
        for cm in corpers:
            work_days = _working_days(cm.user, start, end)
            work_set = set(work_days)
            cm_logs = logs_by_acc.get(getattr(cm, 'account_id', None), [])
            present_dates = set([lg.date for lg in cm_logs]) & work_set
            present = len(present_dates)
            late = 0
            if late_time:
                for lg in cm_logs:
                    if lg.date in work_set and lg.time_in and lg.time_in > late_time:
                        late += 1
            absent = max(0, len(work_days) - present)
            is_first = not WalletTransaction.objects.filter(type='DEBIT', reference__startswith=f"NYSC-{cm.state_code}-").exists()
            override = ClearanceOverride.objects.filter(corper=cm, year_month=yyyymm).exists()
            exceeded_abs = (max_absent is not None and absent > (max_absent or 0))
            exceeded_late = (max_late is not None and late > (max_late or 0))
            qualified = is_first or override or (not exceeded_abs and not exceeded_late)
            downloaded = WalletTransaction.objects.filter(type='DEBIT', reference=f"NYSC-{cm.state_code}-{yyyymm}").exists()
            results.append({
                'id': cm.id,
                'full_name': cm.full_name,
                'state_code': cm.state_code,
                'branch': getattr(cm.branch, 'name', ''),
                'absent': absent,
                'late': late,
                'qualified': qualified,
                'downloaded': downloaded,
                'override': override,
                'reference': f"NYSC-{cm.state_code}-{yyyymm}",
            })
        return Response(results)


class ClearanceApproveView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user = request.user
        role = getattr(user, 'role', None)
        if role not in ('ORG', 'BRANCH'):
            raise PermissionDenied('Only organization or branch admins can approve clearance')
        cm_id = (request.data or {}).get('corper')
        if not cm_id:
            return Response({'detail': 'corper id required'}, status=400)
        cm = CorpMember.objects.filter(id=cm_id).select_related('branch').first()
        if not cm:
            return Response({'detail': 'corper not found'}, status=404)
        if role == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            if not b or cm.branch_id != b.id:
                raise PermissionDenied('Corper does not belong to your branch')
        elif role == 'ORG':
            if cm.user_id != user.id:
                raise PermissionDenied('Corper does not belong to your organization')
        start, end = _prev_month_bounds()
        yyyymm = start.strftime('%Y%m')
        obj, _ = ClearanceOverride.objects.get_or_create(corper=cm, year_month=yyyymm, defaults={'created_by': user})
        return Response({'status': 'approved', 'corper': cm.id, 'year_month': yyyymm})


class WalletFundView(APIView):
    def post(self, request):
        user = request.user
        # Allow only org, or branch admin acting for their org
        if getattr(user, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            user = b.user if b else user
        if getattr(user, 'role', None) != 'ORG':
            raise PermissionDenied('Only organization can fund wallet')
        # Placeholder legacy endpoint
        return Response({'detail': 'Use /api/auth/wallet/paystack/initialize and verify'}, status=202)


class PaystackInitializeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            from .utils import get_paystack_keys
            keys = get_paystack_keys()
        except Exception as e:
            return Response({'detail': str(e)}, status=400)

        data = request.data or {}
        email = data.get('email') or getattr(request.user, 'email', '')
        amount = data.get('amount')
        if not email or not amount:
            return Response({'detail': 'email and amount are required'}, status=400)
        try:
            amt_kobo = int(Decimal(str(amount)) * 100)
        except Exception:
            return Response({'detail': 'invalid amount'}, status=400)
        payload = {
            'email': email,
            'amount': amt_kobo,
        }
        cb = data.get('callback_url')
        if cb:
            payload['callback_url'] = cb
        headers = {
            'Authorization': f"Bearer {keys['secret_key']}",
            'Content-Type': 'application/json'
        }
        try:
            r = requests.post('https://api.paystack.co/transaction/initialize', json=payload, headers=headers, timeout=20)
            jr = r.json()
        except Exception:
            return Response({'detail': 'Failed to reach Paystack'}, status=502)
        if r.status_code != 200 or not jr.get('status'):
            return Response({'detail': jr.get('message', 'Initialization failed')}, status=400)
        pdata = jr.get('data') or {}
        return Response({'authorization_url': pdata.get('authorization_url'), 'reference': pdata.get('reference')})


class PaystackVerifyView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        ref = (request.data or {}).get('reference')
        if not ref:
            return Response({'detail': 'reference is required'}, status=400)
        try:
            from .utils import get_paystack_keys
            keys = get_paystack_keys()
        except Exception as e:
            return Response({'detail': str(e)}, status=400)

        headers = {
            'Authorization': f"Bearer {keys['secret_key']}",
            'Content-Type': 'application/json'
        }
        try:
            r = requests.get(f'https://api.paystack.co/transaction/verify/{ref}', headers=headers, timeout=20)
            jr = r.json()
        except Exception:
            return Response({'detail': 'Failed to reach Paystack'}, status=502)
        if r.status_code != 200 or not jr.get('status'):
            return Response({'detail': jr.get('message', 'Verification failed')}, status=400)
        data = jr.get('data') or {}
        status_val = data.get('status')
        if status_val != 'success':
            return Response({'status': 'failed'}, status=200)
        # Credit wallet on success using customer email and paid amount
        cust = (data.get('customer') or {})
        email = cust.get('email') or getattr(request.user, 'email', None)
        paid_kobo = data.get('amount') or 0
        try:
            paid = (Decimal(paid_kobo) / Decimal('100')).quantize(Decimal('0.01'))
        except Exception:
            paid = Decimal('0.00')
        if not email or paid <= 0:
            return Response({'detail': 'Missing payment data'}, status=400)
        # Credit the requestor's wallet (ORG/BRANCH/CORPER). Fallback to email match if needed.
        target_user = request.user
        if getattr(target_user, 'role', None) not in ('ORG', 'BRANCH', 'CORPER'):
            from .models import OrganizationUser
            target_user = OrganizationUser.objects.filter(email=email).first() or target_user
        acct = _ensure_wallet_with_welcome(target_user)
        WalletTransaction.objects.create(
            account=acct,
            type='CREDIT',
            amount=paid,
            vat_amount=Decimal('0.00'),
            total_amount=paid,
            description='Wallet funding via Paystack',
            reference=str(ref)[:64]
        )
        acct.balance = (acct.balance or Decimal('0.00')) + paid
        acct.save(update_fields=['balance'])
        return Response({'status': 'success', 'balance': str(acct.balance)})
