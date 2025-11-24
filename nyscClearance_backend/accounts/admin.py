from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import (
    OrganizationUser,
    OrganizationProfile,
    BranchOffice,
    Department,
    Unit,
    CorpMember,
    AttendanceLog,
    PublicHoliday,
    LeaveRequest,
    Notification,
    WalletAccount,
    WalletTransaction,
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
    list_display = ('name', 'branch')
    search_fields = ('name', 'branch__name')


@admin.register(Unit)
class UnitAdmin(admin.ModelAdmin):
    list_display = ('name', 'department')
    search_fields = ('name', 'department__name')


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
        total_credit = qs.filter(type='CREDIT').aggregate(s=Sum('total_amount'))['s'] or 0
        total_debit = qs.filter(type='DEBIT').aggregate(s=Sum('total_amount'))['s'] or 0
        response.context_data['total_credit'] = total_credit
        response.context_data['total_debit'] = total_debit
        return response
