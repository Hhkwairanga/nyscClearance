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
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponseNotFound
from django.utils import timezone
from django.utils.decorators import method_decorator
import base64
import math
import numpy as np
import cv2
import os
import shutil
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
from .models import OrganizationProfile, BranchOffice, Department, Unit, CorpMember, PublicHoliday, LeaveRequest, Notification, AttendanceLog
from django.db.models import Count
from django.db import models


User = get_user_model()

# In-memory capture state per corper_id
_CAPTURE_STATE = {}
_ATTENDANCE_STATE = {}

def _reset_capture_state(corper_id: int):
    _CAPTURE_STATE[corper_id] = {
        'image_count': 0,
    }

def _reset_attendance_state(corper_id: int):
    _ATTENDANCE_STATE[corper_id] = {
        'hits': 0,
    }

def _process_capture_frame(corper_id: int, b64_frame: str, save_dir: str):
    st = _CAPTURE_STATE.setdefault(corper_id, {'image_count': 0})
    img_data = base64.b64decode(b64_frame)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, st['image_count']
    # Detect on grayscale for speed/robustness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    # Save whenever a face is detected, up to 100 images
    if len(faces) > 0 and st['image_count'] < 100:
        os.makedirs(save_dir, exist_ok=True)
        # take the first detected face per frame to avoid oversampling
        (x,y,w,h) = faces[0]
        # Use color ROI for downstream encoding (RGB pipeline)
        face_roi_color = img[y:y+h, x:x+w]
        if face_roi_color.size != 0:
            # Resize to a modest size to reduce disk + CPU
            resized_color = cv2.resize(face_roi_color, (160, 160))
            out_path = os.path.join(save_dir, f"frame_{st['image_count']}.jpg")
            # Save as color JPEG (BGR order expected by imwrite)
            cv2.imwrite(out_path, resized_color)
            st['image_count'] += 1

    # draw overlays
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, f'Image {st["image_count"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', img)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    return processed_frame, st['image_count']


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
    frontend_base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', 'http://localhost:5173')).rstrip('/')

    ctx = {
        'corper_id': corper.id,
        'full_name': corper.full_name,
        'state_code': corper.state_code,
        'frontend_base': frontend_base,
    }
    return render(request, 'capture.html', ctx)


@csrf_exempt
def capture_process_frame(request, corper_id: int):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    frame = request.POST.get('frame')
    if not frame:
        return JsonResponse({'detail': 'Missing frame'}, status=400)
    # Save dir per corper under media/captures/{id}
    media_root = getattr(settings, 'MEDIA_ROOT', os.path.join(os.getcwd(), 'media'))
    save_dir = os.path.join(media_root, 'captures', str(corper_id))
    processed, count = _process_capture_frame(corper_id, frame, save_dir)
    if not processed:
        return JsonResponse({'detail': 'Processing failed'}, status=500)
    return JsonResponse({'frame': processed, 'saved': count})


@csrf_exempt
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

    media_root = getattr(settings, 'MEDIA_ROOT', os.path.join(os.getcwd(), 'media'))
    save_dir = os.path.join(media_root, 'captures', str(corper_id))
    if not os.path.isdir(save_dir):
        return JsonResponse({'detail': 'No captured images'}, status=400)

    # Load up to 100 images and compute encodings
    files = sorted([f for f in os.listdir(save_dir) if f.lower().endswith('.jpg')])
    encodings = []
    for fname in files[:100]:
        path = os.path.join(save_dir, fname)
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                encodings.append(encs[0])
        except Exception:
            continue

    if not encodings:
        return JsonResponse({'detail': 'No encodings found; ensure face is visible'}, status=400)

    # Average encoding
    avg = np.mean(np.stack(encodings, axis=0), axis=0)
    # Save to model as JSON list of floats
    try:
        import json
        corper.face_encoding = json.dumps(avg.tolist())
        corper.save(update_fields=['face_encoding'])
    except Exception as e:
        return JsonResponse({'detail': f'Failed to save encoding: {e}'}, status=500)

    # Cleanup captured images
    try:
        shutil.rmtree(save_dir)
    except Exception:
        pass

    # Reset in-memory state
    _reset_capture_state(corper_id)

    return JsonResponse({'status': 'ok', 'encodings': len(encodings)})


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
    frontend_base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', 'http://localhost:5173')).rstrip('/')

    ctx = {
        'corper_id': cm.id,
        'full_name': cm.full_name,
        'state_code': cm.state_code,
        'frontend_base': frontend_base,
    }
    return render(request, 'attendance.html', ctx)


def _get_target_location_for_corper(cm):
    """Return (lat, lng, source) for the corper's expected location.
    If both branch and org locations exist, use their midpoint as an "average" center.
    """
    br_lat = br_lng = None
    org_lat = org_lng = None
    try:
        br = cm.branch
        if br and br.latitude is not None and br.longitude is not None:
            br_lat, br_lng = br.latitude, br.longitude
    except Exception:
        pass
    try:
        prof = OrganizationProfile.objects.filter(user=cm.user).first()
        if prof and prof.location_lat is not None and prof.location_lng is not None:
            org_lat, org_lng = prof.location_lat, prof.location_lng
    except Exception:
        pass
    if br_lat is not None and org_lat is not None:
        # Midpoint (approx) as average geofence center
        lat = (br_lat + org_lat) / 2.0
        lng = (br_lng + org_lng) / 2.0
        return lat, lng, 'avg'
    if br_lat is not None:
        return br_lat, br_lng, 'branch'
    if org_lat is not None:
        return org_lat, org_lng, 'org'
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


@csrf_exempt
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
    return JsonResponse({'allowed': allowed, 'distance_m': round(dist, 1), 'threshold_m': threshold, 'source': src or 'unknown'})


@csrf_exempt
def attendance_process_frame(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    user = request.user
    if getattr(user, 'role', None) != 'CORPER':
        return JsonResponse({'detail': 'Not allowed'}, status=403)
    cm = getattr(user, 'corper_profile', None)
    if not cm:
        return JsonResponse({'detail': 'Corper profile not found'}, status=404)
    frame = request.POST.get('frame')
    if not frame:
        return JsonResponse({'detail': 'Missing frame'}, status=400)

    # Load stored encoding
    import json
    try:
        saved = np.array(json.loads(cm.face_encoding), dtype='float32') if cm.face_encoding else None
    except Exception:
        saved = None
    if saved is None:
        return JsonResponse({'detail': 'No face encoding on file'}, status=400)

    # Decode incoming frame and detect face(s)
    img_data = base64.b64decode(frame)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JsonResponse({'detail': 'Invalid frame'}, status=400)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect faces and compute encodings with locations to draw overlays
    try:
        face_locations = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, face_locations)
    except Exception:
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
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0) if is_match else (0, 0, 255), 2)
        # Grey background bar
        cv2.rectangle(img, (right + 10, top), (right + 30, bottom), (128, 128, 128), -1)
        # Filled confidence portion (green if >=65 else red)
        bar_height = bottom - top
        filled = int(bar_height * (conf / 100.0))
        color = (0, 255, 0) if conf >= 65 else (0, 0, 255)
        cv2.rectangle(img, (right + 10, bottom - filled), (right + 30, bottom), color, -1)
        # If confident, show name + state_code box
        if conf >= 65:
            cv2.rectangle(img, (right + 35, top), (right + 260, top + 45), (128, 128, 128), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, f"{cm.full_name}", (right + 40, top + 40), font, 0.6, (0, 255, 0), 1)
            cv2.putText(img, f"{cm.state_code}", (right + 40, top + 20), font, 0.6, (0, 255, 0), 1)

    # Update recognition state using the best face confidence
    st = _ATTENDANCE_STATE.setdefault(cm.id, {'hits': 0})
    if best_conf >= 65:  # ~0.35 distance equivalence; empirical
        st['hits'] = min(10, st.get('hits', 0) + 1)
        if st['hits'] >= 3:
            recognized = True
    else:
        st['hits'] = 0

    # Header label
    label = 'RECOGNIZED' if recognized else 'Scanning…'
    head_color = (0, 200, 0) if recognized else (0, 200, 200)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_color, 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', img)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    return JsonResponse({'frame': processed_frame, 'recognized': recognized})


@csrf_exempt
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
    # Persist attendance log: create/update today's record
    from .models import AttendanceLog
    today = timezone.localdate()
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
    # If created, set time_in; else, fill missing time_in or time_out accordingly
    if created or not log.time_in:
        log.time_in = now.time()
    elif not log.time_out:
        # set time_out only if logically after time_in
        if not log.time_in or now.time() >= log.time_in:
            log.time_out = now.time()
    log.save()

    _reset_attendance_state(cm.id)
    return JsonResponse({'status': 'ok'})


class RegisterView(APIView):
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

        # Redirect to frontend page. If user has no usable password, include token for password setup
        # Prefer request Origin if it matches configured FRONTEND_ORIGINS for better DX (5173/5174)
        request_origin = request.headers.get('Origin')
        try:
            allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
        except Exception:
            allowed = set()
        base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', 'http://localhost:5173')).rstrip('/')
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
        base = getattr(settings, 'FRONTEND_ORIGIN', 'http://localhost:5173').rstrip('/')
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


class LoginView(APIView):
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
        login(request, user)
        OrganizationProfile.objects.get_or_create(user=user)
        return Response({'message': 'Logged in'})


class LogoutView(APIView):
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
            'attendance': self._attendance_stats(att_qs)
        }
        return Response(data)

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
