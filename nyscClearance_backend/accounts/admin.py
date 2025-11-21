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
)


@admin.register(OrganizationUser)
class OrganizationUserAdmin(BaseUserAdmin):
    ordering = ('email',)
    list_display = ('email', 'name', 'number_of_corpers', 'is_active', 'is_email_verified')
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('name', 'address', 'number_of_corpers')}),
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
    list_display = ('name', 'user', 'address')
    search_fields = ('name', 'user__email')


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
