from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import (
    OrganizationUser,
    OrganizationProfile,
    BranchOffice,
    Department,
    Unit,
    CorpMember,
    TempFaceEncoding,
    AttendanceLog,
    PublicHoliday,
    NationalHoliday,
    LeaveRequest,
    Notification,
    WalletAccount,
    WalletTransaction,
    SystemSetting,
    PaystackConfig,
    GoogleMapsConfig,
    SubscriptionPlanSetting,
    OrganizationSubscription,
    SubscriptionPayment,
    ClearanceAccess,
    ClearanceOverride,
)


@admin.register(OrganizationUser)
class OrganizationUserAdmin(BaseUserAdmin):
    ordering = ('email',)
    list_display = ('email', 'name', 'phone_number', 'is_active', 'is_email_verified')
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('name', 'address', 'phone_number')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'is_email_verified', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login',)}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'name', 'password1', 'password2', 'is_staff', 'is_superuser'),
        }),
    )
    search_fields = ('email', 'name')
    filter_horizontal = ('groups', 'user_permissions',)


@admin.register(OrganizationProfile)
class OrganizationProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'late_time', 'closing_time', 'max_days_late', 'max_days_absent')


@admin.register(BranchOffice)
class BranchOfficeAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'admin', 'address')
    search_fields = ('name', 'user__email', 'admin__email')


@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ('name', 'user')
    search_fields = ('name', 'user__email')


@admin.register(Unit)
class UnitAdmin(admin.ModelAdmin):
    list_display = ('name', 'user')
    search_fields = ('name', 'user__email')


@admin.register(CorpMember)
class CorpMemberAdmin(admin.ModelAdmin):
    list_display = ('full_name', 'state_code', 'gender', 'user', 'passing_out_date')
    search_fields = ('full_name', 'state_code', 'user__email')


@admin.register(AttendanceLog)
class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'date', 'time_in', 'time_out', 'org', 'account')
    list_filter = ('date', 'org')
    search_fields = ('name', 'code', 'account__email', 'org__email')


@admin.register(PublicHoliday)
class PublicHolidayAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'start_date', 'end_date', 'created_at')
    list_filter = ('user', 'start_date', 'end_date')
    search_fields = ('title', 'user__email')


@admin.register(NationalHoliday)
class NationalHolidayAdmin(admin.ModelAdmin):
    list_display = ('country_code', 'date', 'name')
    list_filter = ('country_code', 'date')
    search_fields = ('name', 'local_name', 'country_code')


@admin.register(LeaveRequest)
class LeaveRequestAdmin(admin.ModelAdmin):
    list_display = ('corper', 'branch', 'start_date', 'end_date', 'status', 'decided_by', 'created_at')
    list_filter = ('status', 'branch', 'start_date', 'end_date')
    search_fields = ('corper__full_name', 'corper__state_code', 'branch__name', 'decided_by__email')


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'branch', 'created_by', 'created_at')
    list_filter = ('user', 'branch')
    search_fields = ('title', 'user__email', 'branch__name', 'created_by__email')


@admin.register(WalletAccount)
class WalletAccountAdmin(admin.ModelAdmin):
    list_display = ('user', 'balance', 'created_at')
    search_fields = ('user__email',)


@admin.register(WalletTransaction)
class WalletTransactionAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'account', 'type', 'description', 'reference', 'amount', 'vat_amount', 'total_amount')
    list_filter = ('type', 'account')
    search_fields = ('description', 'reference', 'account__user__email')

    def changelist_view(self, request, extra_context=None):
        response = super().changelist_view(request, extra_context=extra_context)
        try:
            cl = response.context_data['cl']
            qs = cl.queryset
        except Exception:
            return response
        from django.db.models import Sum
        # Compute totals for the entire transactions table (ignoring pagination and filters)
        all_qs = WalletTransaction.objects.all()
        total_credit_all = all_qs.filter(type='CREDIT').aggregate(s=Sum('total_amount'))['s'] or 0
        total_debit_all = all_qs.filter(type='DEBIT').aggregate(s=Sum('total_amount'))['s'] or 0

        # Keep page/filtered totals as well (using cl.queryset which may be limited)
        total_credit = qs.filter(type='CREDIT').aggregate(s=Sum('total_amount'))['s'] or 0
        total_debit = qs.filter(type='DEBIT').aggregate(s=Sum('total_amount'))['s'] or 0

        response.context_data['total_credit_all'] = total_credit_all
        response.context_data['total_debit_all'] = total_debit_all
        response.context_data['total_credit'] = total_credit
        response.context_data['total_debit'] = total_debit
        return response


@admin.register(SystemSetting)
class SystemSettingAdmin(admin.ModelAdmin):
    list_display = ('welcome_bonus', 'clearance_fee', 'discount_enabled', 'discount_percent', 'notify_enabled', 'auth_token_version', 'updated_at')
    fieldsets = (
        ('Wallet', {
            'fields': ('welcome_bonus', 'clearance_fee')
        }),
        ('Discounts', {
            'fields': ('discount_enabled', 'discount_percent')
        }),
        ('Announcement', {
            'fields': ('notify_enabled', 'notify_title', 'notify_message', 'notify_start', 'notify_end')
        }),
        ('Security', {
            'fields': ('auth_token_version',),
            'description': 'Increment this value to invalidate all existing bearer tokens. Use the force_logout management command for full session logout.',
        }),
    )

    def has_add_permission(self, request):
        # Allow only a single settings row
        from .models import SystemSetting
        if SystemSetting.objects.exists():
            return False
        return super().has_add_permission(request)

    def has_delete_permission(self, request, obj=None):
        # Prevent deletion to keep singleton semantics
        return False


@admin.register(PaystackConfig)
class PaystackConfigAdmin(admin.ModelAdmin):
    list_display = ("public_key", "is_active", "updated_at")
    search_fields = ("public_key",)


@admin.register(GoogleMapsConfig)
class GoogleMapsConfigAdmin(admin.ModelAdmin):
    list_display = ("name", "is_active", "has_browser_key", "has_server_key", "map_id", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "map_id")

    def has_browser_key(self, obj):
        return bool(obj.browser_api_key)
    has_browser_key.boolean = True

    def has_server_key(self, obj):
        return bool(obj.server_api_key)
    has_server_key.boolean = True


@admin.register(SubscriptionPlanSetting)
class SubscriptionPlanSettingAdmin(admin.ModelAdmin):
    list_display = (
        "sort_order",
        "name",
        "code",
        "corper_min",
        "corper_max",
        "monthly_price",
        "yearly_price",
        "discount_enabled",
        "discount_percent",
        "is_active",
    )
    list_editable = ("monthly_price", "yearly_price", "discount_enabled", "discount_percent", "is_active")
    list_filter = ("is_active", "discount_enabled")
    search_fields = ("name", "code")


@admin.register(OrganizationSubscription)
class OrganizationSubscriptionAdmin(admin.ModelAdmin):
    list_display = ("org", "plan_name", "billing_cycle", "status", "amount_paid", "starts_at", "expires_at")
    list_filter = ("status", "billing_cycle", "plan_code")
    search_fields = ("org__email", "org__name", "plan_name")


@admin.register(SubscriptionPayment)
class SubscriptionPaymentAdmin(admin.ModelAdmin):
    list_display = ("created_at", "org", "plan_name", "billing_cycle", "status", "amount_charged", "reference", "paid_at")
    list_filter = ("status", "billing_cycle", "plan_code")
    search_fields = ("org__email", "org__name", "reference", "plan_name")
    readonly_fields = ("raw_response", "created_at", "updated_at", "paid_at")


@admin.register(ClearanceAccess)
class ClearanceAccessAdmin(admin.ModelAdmin):
    list_display = ("created_at", "corper", "year_month", "source", "org", "branch", "reference")
    list_filter = ("source", "year_month", "org")
    search_fields = ("corper__full_name", "corper__state_code", "reference", "org__email", "branch__name")


@admin.register(ClearanceOverride)
class ClearanceOverrideAdmin(admin.ModelAdmin):
    list_display = ("corper", "year_month", "created_by", "created_at")
    search_fields = ("corper__full_name", "corper__state_code", "year_month")


@admin.register(TempFaceEncoding)
class TempFaceEncodingAdmin(admin.ModelAdmin):
    list_display = ("corper", "session_id", "idx", "created_at")
    list_filter = ("session_id", "created_at")
    search_fields = ("corper__full_name", "corper__state_code", "session_id")
    readonly_fields = ("created_at",)
