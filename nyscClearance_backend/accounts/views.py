from django.contrib.auth import get_user_model, authenticate, login, logout
from django.conf import settings
from django.shortcuts import redirect
from rest_framework import status, permissions, viewsets
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import action

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
from .models import OrganizationProfile, BranchOffice, Department, Unit, CorpMember, PublicHoliday, LeaveRequest, Notification
from django.db.models import Count
from django.db import models


User = get_user_model()


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
        base = getattr(settings, 'FRONTEND_ORIGIN', 'http://localhost:5173').rstrip('/')
        # If the user has no password (invited admin/corper), send to password set page with token
        role = request.query_params.get('role') or getattr(user, 'role', None)
        if not user.has_usable_password():
            return redirect(f"{base}/verify-success?token={token}{f'&role={role}' if role else ''}")
        # Otherwise, show success page (no password needed)
        return redirect(f"{base}/verify-success")


class PasswordSetView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        token = request.data.get('token')
        password = request.data.get('password')
        if not token or not password:
            return Response({'detail': 'token and password are required'}, status=status.HTTP_400_BAD_REQUEST)
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
        if not token or not password:
            return Response({'detail': 'token and password are required'}, status=status.HTTP_400_BAD_REQUEST)
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
        return Response(status=status.HTTP_204_NO_CONTENT)


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
        return BranchOffice.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # user is set inside the serializer using request from context
        serializer.save()


class DepartmentViewSet(viewsets.ModelViewSet):
    serializer_class = DepartmentSerializer
    def get_queryset(self):
        return Department.objects.filter(branch__user=self.request.user)

    def perform_create(self, serializer):
        # ensure branch belongs to user
        branch = serializer.validated_data.get('branch')
        if branch.user_id != self.request.user.id:
            raise PermissionDenied('Invalid branch')
        serializer.save()


class UnitViewSet(viewsets.ModelViewSet):
    serializer_class = UnitSerializer
    def get_queryset(self):
        return Unit.objects.filter(department__branch__user=self.request.user)

    def perform_create(self, serializer):
        dept = serializer.validated_data.get('department')
        if dept.branch.user_id != self.request.user.id:
            raise PermissionDenied('Invalid department')
        serializer.save()


class CorpMemberViewSet(viewsets.ModelViewSet):
    serializer_class = CorpMemberSerializer

    def get_queryset(self):
        user = self.request.user
        if user.role == 'ORG':
            return CorpMember.objects.filter(user=user)
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
        # user is set inside the serializer using request from context
        serializer.save()


class PublicHolidayViewSet(viewsets.ModelViewSet):
    serializer_class = PublicHolidaySerializer

    def get_queryset(self):
        user = self.request.user
        if getattr(user, 'role', 'ORG') == 'ORG':
            return PublicHoliday.objects.filter(user=user).order_by('date')
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
        return PublicHoliday.objects.filter(user=org_user).order_by('date')

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
        branches = BranchOffice.objects.filter(user=request.user)
        corpers = CorpMember.objects.filter(user=request.user)
        departments = Department.objects.filter(branch__user=request.user)
        units = Unit.objects.filter(department__branch__user=request.user)

        by_branch_qs = (
            corpers.values('branch__name')
                   .annotate(count=Count('id'))
                   .order_by('branch__name')
        )
        corpers_by_branch = []
        for row in by_branch_qs:
            name = row['branch__name'] or 'Unassigned'
            corpers_by_branch.append({'branch': name, 'count': row['count']})

        data = {
            'totals': {
                'branches': branches.count(),
                'departments': departments.count(),
                'units': units.count(),
                'corpers': corpers.count(),
            },
            'corpers_by_branch': corpers_by_branch,
            'attendance': {
                'today': 0,
                'this_month': 0,
            }
        }
        return Response(data)


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
