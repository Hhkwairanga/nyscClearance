from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    RegisterView,
    VerifyEmailView,
    CSRFView,
    ConfigView,
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
    QueryRecordViewSet,
    capture_page,
    capture_process_frame,
    capture_finalize,
    attendance_page,
    attendance_process_frame,
    attendance_finalize,
    attendance_authorize,
    performance_summary,
    performance_clearance_page,
    WalletView,
    WalletStatementExportView,
    WalletFundView,
    wallet_charge_clearance,
    AnnouncementView,
    PaystackInitializeView,
    PaystackVerifyView,
    PaystackWebhookView,
    SubscriptionPlansView,
    SubscriptionStatusView,
    SubscriptionInitializeView,
    SubscriptionVerifyView,
    ClearanceStatusView,
    ClearanceApproveView,
    AllHolidaysView,
    AttendanceReportView,
    CorperAttendanceReportView,
    AttendanceLogReportView,
    AttendanceExcelExportView,
)

router = DefaultRouter()
router.register(r'branches', BranchOfficeViewSet, basename='branch')
router.register(r'departments', DepartmentViewSet, basename='department')
router.register(r'units', UnitViewSet, basename='unit')
router.register(r'corpers', CorpMemberViewSet, basename='corper')
router.register(r'holidays', PublicHolidayViewSet, basename='holiday')
router.register(r'leaves', LeaveRequestViewSet, basename='leave')
router.register(r'notifications', NotificationViewSet, basename='notification')
router.register(r'queries', QueryRecordViewSet, basename='query')

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('verify/', VerifyEmailView.as_view(), name='verify-email'),
    path('password/set/', PasswordSetView.as_view(), name='password-set'),
    path('password/reset/', PasswordResetRequestView.as_view(), name='password-reset'),
    path('password/reset/confirm/', PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
    path('csrf/', CSRFView.as_view(), name='csrf'),
    path('config/', ConfigView.as_view(), name='config'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('me/', MeView.as_view(), name='me'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('stats/', StatsView.as_view(), name='stats'),
    path('holidays/all/', AllHolidaysView.as_view(), name='holidays-all'),
    path('reports/attendance/', AttendanceReportView.as_view(), name='report-attendance'),
    path('reports/corpers/', CorperAttendanceReportView.as_view(), name='report-corpers'),
    path('reports/attendance/logs/', AttendanceLogReportView.as_view(), name='report-attendance-logs'),
    path('reports/attendance/export/', AttendanceExcelExportView.as_view(), name='report-attendance-export'),
    # Face capture (HTML page + processing endpoint)
    path('capture/<int:corper_id>/', capture_page, name='capture'),
    path('capture/<int:corper_id>/process/', capture_process_frame, name='capture-process'),
    path('capture/<int:corper_id>/finalize/', capture_finalize, name='capture-finalize'),
    # Attendance (self-service for corpers)
    path('attendance/', attendance_page, name='attendance'),
    path('attendance/authorize/', attendance_authorize, name='attendance-authorize'),
    path('attendance/process/', attendance_process_frame, name='attendance-process'),
    path('attendance/finalize/', attendance_finalize, name='attendance-finalize'),
    # Performance clearance (corper)
    path('performance/summary/', performance_summary, name='performance-summary'),
    path('performance/clearance/', performance_clearance_page, name='performance-clearance'),
    # Wallet
    path('wallet/', WalletView.as_view(), name='wallet'),
    path('wallet/export/', WalletStatementExportView.as_view(), name='wallet-export'),
    path('wallet/fund/', WalletFundView.as_view(), name='wallet-fund'),
    path('wallet/charge_clearance/', wallet_charge_clearance, name='wallet-charge-clearance'),
    # Paystack
    path('wallet/paystack/initialize/', PaystackInitializeView.as_view(), name='paystack-initialize'),
    path('wallet/paystack/verify/', PaystackVerifyView.as_view(), name='paystack-verify'),
    path('paystack/webhook/', PaystackWebhookView.as_view(), name='paystack-webhook'),
    # Subscriptions
    path('subscriptions/plans/', SubscriptionPlansView.as_view(), name='subscription-plans'),
    path('subscriptions/status/', SubscriptionStatusView.as_view(), name='subscription-status'),
    path('subscriptions/initialize/', SubscriptionInitializeView.as_view(), name='subscription-initialize'),
    path('subscriptions/verify/', SubscriptionVerifyView.as_view(), name='subscription-verify'),
    # Clearance admin
    path('clearance/status/', ClearanceStatusView.as_view(), name='clearance-status'),
    path('clearance/approve/', ClearanceApproveView.as_view(), name='clearance-approve'),
    # System announcement for org dashboard
    path('announcement/', AnnouncementView.as_view(), name='announcement'),
    path('', include(router.urls)),
]
