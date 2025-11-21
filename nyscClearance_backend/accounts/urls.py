from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    RegisterView,
    VerifyEmailView,
    CSRFView,
    LoginView,
    LogoutView,
    MeView,
    ProfileView,
    StatsView,
    PasswordSetView,
    PasswordResetRequestView,
    PasswordResetConfirmView,
    BranchOfficeViewSet,
    DepartmentViewSet,
    UnitViewSet,
    CorpMemberViewSet,
    PublicHolidayViewSet,
    LeaveRequestViewSet,
    NotificationViewSet,
    capture_page,
    capture_process_frame,
    capture_finalize,
    attendance_page,
    attendance_process_frame,
    attendance_finalize,
)

router = DefaultRouter()
router.register(r'branches', BranchOfficeViewSet, basename='branch')
router.register(r'departments', DepartmentViewSet, basename='department')
router.register(r'units', UnitViewSet, basename='unit')
router.register(r'corpers', CorpMemberViewSet, basename='corper')
router.register(r'holidays', PublicHolidayViewSet, basename='holiday')
router.register(r'leaves', LeaveRequestViewSet, basename='leave')
router.register(r'notifications', NotificationViewSet, basename='notification')

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('verify/', VerifyEmailView.as_view(), name='verify-email'),
    path('password/set/', PasswordSetView.as_view(), name='password-set'),
    path('password/reset/', PasswordResetRequestView.as_view(), name='password-reset'),
    path('password/reset/confirm/', PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
    path('csrf/', CSRFView.as_view(), name='csrf'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('me/', MeView.as_view(), name='me'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('stats/', StatsView.as_view(), name='stats'),
    # Face capture (HTML page + processing endpoint)
    path('capture/<int:corper_id>/', capture_page, name='capture'),
    path('capture/<int:corper_id>/process/', capture_process_frame, name='capture-process'),
    path('capture/<int:corper_id>/finalize/', capture_finalize, name='capture-finalize'),
    # Attendance (self-service for corpers)
    path('attendance/', attendance_page, name='attendance'),
    path('attendance/process/', attendance_process_frame, name='attendance-process'),
    path('attendance/finalize/', attendance_finalize, name='attendance-finalize'),
    path('', include(router.urls)),
]
