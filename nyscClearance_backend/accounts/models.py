"""
Accounts app models

Core entities:
- OrganizationUser: single user model (roles: ORG, BRANCH, CORPER)
- OrganizationProfile: org metadata (logo, signatory, times)
- BranchOffice, Department, Unit: organization structure
- CorpMember: enrolled corp members and their placement
- AttendanceLog: per-account daily attendance (time_in, time_out)
- WalletAccount / WalletTransaction: simple wallet and transaction log
- SystemSetting: singleton for welcome bonus, clearance fee, discounts, announcements
- SubscriptionPlanSetting / OrganizationSubscription / SubscriptionPayment:
  database-managed subscription pricing and billing records
- ClearanceAccess: tracks monthly clearance access independent of wallet billing
- PaystackConfig: active Paystack API keys storage (admin-managed)

Conventions:
- Wallet exists for every user; welcome bonus applies only to ORG on first wallet creation
- WalletTransaction.total_amount includes VAT for debits; credits use 0 VAT
"""

from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models
from django.conf import settings
from decimal import Decimal
from datetime import time
from django.utils import timezone
import uuid


class OrganizationUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        return self.create_user(email, password, **extra_fields)


class OrganizationUser(AbstractBaseUser, PermissionsMixin):
    ROLE_CHOICES = (
        ('ORG', 'Organization'),
        ('BRANCH', 'Branch Admin'),
        ('CORPER', 'Corper'),
    )
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255)
    address = models.TextField(blank=True)
    phone_number = models.CharField(max_length=32, blank=True)

    is_active = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    is_email_verified = models.BooleanField(default=False)
    # When true, user must change password before using the app (used for bulk-created admins).
    force_password_change = models.BooleanField(default=False)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='ORG')

    date_joined = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']

    objects = OrganizationUserManager()

    def __str__(self):
        return self.email


class OrganizationProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')
    logo = models.FileField(upload_to='org_logos/', blank=True, null=True)
    # Director Human Resource signatory details
    signatory_name = models.CharField(max_length=255, blank=True)
    signature = models.ImageField(upload_to='org_signatures/', blank=True, null=True)
    late_time = models.TimeField(blank=True, null=True, default=time(8, 30))
    closing_time = models.TimeField(blank=True, null=True, default=time(16, 0))
    max_days_late = models.PositiveSmallIntegerField(blank=True, null=True, default=5)
    max_days_absent = models.PositiveSmallIntegerField(blank=True, null=True, default=3)
    location_lat = models.FloatField(blank=True, null=True)
    location_lng = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"Profile for {self.user.email}"


class BranchOffice(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='branches')
    name = models.CharField(max_length=255)
    address = models.TextField(blank=True)
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    # Optional branch admin linkage and details
    admin = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='managed_branches'
    )
    admin_staff_id = models.CharField(max_length=64, blank=True)

    def __str__(self):
        return f"{self.name} ({self.user.email})"


class PublicHoliday(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='holidays')
    title = models.CharField(max_length=255)
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'start_date', 'end_date')

    def __str__(self):
        return f"{self.title} - {self.start_date} to {self.end_date}"


class NationalHoliday(models.Model):
    """Country-level public holidays synced from Nager.Date.

    These are read-only from the UI and apply to all organisations.
    Organisations can still add custom holidays via `PublicHoliday`.
    """

    country_code = models.CharField(max_length=2, default='NG', db_index=True)
    date = models.DateField(db_index=True)
    name = models.CharField(max_length=255)
    local_name = models.CharField(max_length=255, blank=True, default='')
    raw = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('country_code', 'date', 'name')
        ordering = ('date', 'name')

    def __str__(self):
        return f"{self.country_code} {self.date} {self.name}"


class LeaveRequest(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
    )
    corper = models.ForeignKey('accounts.CorpMember', on_delete=models.CASCADE, related_name='leave_requests')
    branch = models.ForeignKey(BranchOffice, on_delete=models.SET_NULL, null=True, blank=True, related_name='leave_requests')
    start_date = models.DateField()
    end_date = models.DateField()
    reason = models.TextField(blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    decided_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='decided_leaves')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.corper.full_name} {self.start_date}->{self.end_date} [{self.status}]"


class Notification(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    branch = models.ForeignKey(BranchOffice, on_delete=models.CASCADE, null=True, blank=True, related_name='notifications')
    title = models.CharField(max_length=255)
    message = models.TextField()
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='created_notifications')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        scope = 'All' if not self.branch else f'Branch {self.branch.name}'
        return f"{self.title} ({scope})"


class QueryRecord(models.Model):
    """Disciplinary/HR queries issued to a corper."""

    STATUS_CHOICES = (
        ('OPEN', 'Open'),
        ('RESOLVED', 'Resolved'),
    )

    org = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='queries')
    branch = models.ForeignKey(BranchOffice, on_delete=models.SET_NULL, null=True, blank=True, related_name='queries')
    corper = models.ForeignKey('accounts.CorpMember', on_delete=models.CASCADE, related_name='queries')
    title = models.CharField(max_length=255)
    message = models.TextField(blank=True)
    corper_reply = models.TextField(blank=True)
    replied_at = models.DateTimeField(null=True, blank=True)
    replied_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='query_replies',
    )
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='OPEN')
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='created_queries')
    resolved_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='resolved_queries')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('-created_at', '-id')

    def __str__(self):
        return f"{self.title} ({self.corper.state_code})"


class Department(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='departments')
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name} ({self.user.email})"


class Unit(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='units')
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name} ({self.user.email})"


class CorpMember(models.Model):
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='corpers')
    full_name = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    state_code = models.CharField(max_length=12)
    passing_out_date = models.DateField()
    cds_day = models.PositiveSmallIntegerField(blank=True, null=True, help_text='CDS weekday: 0=Mon..4=Fri')
    account = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='corper_profile')
    branch = models.ForeignKey(BranchOffice, on_delete=models.SET_NULL, null=True, blank=True, related_name='corpers')
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True, related_name='corpers')
    unit = models.ForeignKey(Unit, on_delete=models.SET_NULL, null=True, blank=True, related_name='corpers')
    face_encoding = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.full_name} ({self.state_code})"


class TempFaceEncoding(models.Model):
    """Temporary per-frame encodings captured during a session.
    Keyed by corper and a `session_id` so work can be spread across workers.
    Deleted after finalization.
    """
    corper = models.ForeignKey('accounts.CorpMember', on_delete=models.CASCADE, related_name='temp_face_encodings')
    session_id = models.CharField(max_length=64, db_index=True)
    idx = models.PositiveIntegerField(default=0)
    vector = models.TextField()  # JSON list of floats
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['corper', 'session_id']),
        ]
        ordering = ('created_at', 'id')

    def __str__(self):
        return f"TmpEnc corper={self.corper_id} session={self.session_id} idx={self.idx}"

class AttendanceLog(models.Model):
    # The corper's user account (login identity)
    account = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='attendance_logs')
    # Owning organization user (for easy scoping/filtering)
    org = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='org_attendance_logs')
    # Snapshot fields for display/reporting
    name = models.CharField(max_length=255)
    state = models.CharField(max_length=64, blank=True)
    code = models.CharField(max_length=32, blank=True)  # state code

    date = models.DateField()
    time_in = models.TimeField(blank=True, null=True)
    time_out = models.TimeField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('account', 'date')
        ordering = ('-date', '-created_at')

    def __str__(self):
        return f"{self.name} {self.date} in:{self.time_in} out:{self.time_out}"


class WalletAccount(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='wallet')
    balance = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Wallet({self.user.email}) balance={self.balance}"


class WalletTransaction(models.Model):
    TYPE_CHOICES = (
        ('CREDIT', 'Credit'),
        ('DEBIT', 'Debit'),
    )
    account = models.ForeignKey(WalletAccount, on_delete=models.CASCADE, related_name='transactions')
    type = models.CharField(max_length=6, choices=TYPE_CHOICES)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    vat_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    total_amount = models.DecimalField(max_digits=12, decimal_places=2)
    description = models.CharField(max_length=255, blank=True)
    reference = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created_at', '-id')

    def __str__(self):
        return f"{self.type} {self.total_amount} ({self.description})"


class SystemSetting(models.Model):
    """Global system settings editable from Django admin only.

    - welcome_bonus: amount credited when an organization wallet is created
    - discount_enabled/percent: apply percentage discount to charges when enabled
    - notify_*: optional scheduled announcement for organization dashboard
    - auth_token_version: increment to invalidate all previously issued access tokens
    """
    welcome_bonus = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('10000.00'))
    clearance_fee = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('300.00'))
    discount_enabled = models.BooleanField(default=False)
    discount_percent = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))  # 0-100
    notify_enabled = models.BooleanField(default=False)
    notify_title = models.CharField(max_length=200, blank=True)
    notify_message = models.TextField(blank=True)
    notify_start = models.DateTimeField(null=True, blank=True)
    notify_end = models.DateTimeField(null=True, blank=True)
    auth_token_version = models.PositiveIntegerField(default=1)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'System Setting'
        verbose_name_plural = 'System Settings'

    def __str__(self):
        return 'System Settings'

    @classmethod
    def current(cls):
        obj, _ = cls.objects.get_or_create(id=1, defaults={})
        return obj


class PaystackConfig(models.Model):
    public_key = models.CharField(max_length=200)
    secret_key = models.CharField(max_length=200)
    webhook_secret = models.CharField(max_length=200, blank=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Paystack Config'
        verbose_name_plural = 'Paystack Config'

    def __str__(self):
        return f"Paystack ({'active' if self.is_active else 'inactive'})"


class GoogleMapsConfig(models.Model):
    """Google Maps credentials managed from Django admin."""

    name = models.CharField(max_length=120, default='Default Google Maps Config')
    browser_api_key = models.CharField(max_length=255, blank=True)
    server_api_key = models.CharField(max_length=255, blank=True)
    map_id = models.CharField(max_length=120, blank=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Google Maps Config'
        verbose_name_plural = 'Google Maps Config'

    def __str__(self):
        return f"{self.name} ({'active' if self.is_active else 'inactive'})"

    @classmethod
    def active(cls):
        return cls.objects.filter(is_active=True).order_by('-updated_at', '-id').first()


class SubscriptionPlanSetting(models.Model):
    PLAN_CHOICES = (
        ('STARTER', 'Starter'),
        ('BASIC', 'Basic'),
        ('PRO', 'Premium'),
        ('ENTERPRISE', 'Enterprise'),
    )

    code = models.CharField(max_length=20, choices=PLAN_CHOICES, unique=True)
    name = models.CharField(max_length=80)
    corper_min = models.PositiveIntegerField(default=0)
    corper_max = models.PositiveIntegerField(null=True, blank=True)
    monthly_price = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    yearly_price = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    discount_enabled = models.BooleanField(default=False)
    discount_percent = models.DecimalField(max_digits=5, decimal_places=2, default=Decimal('0.00'))
    is_active = models.BooleanField(default=True)
    sort_order = models.PositiveSmallIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('sort_order', 'id')
        verbose_name = 'Subscription Plan Setting'
        verbose_name_plural = 'Subscription Plan Settings'

    def __str__(self):
        return self.name or self.code


class OrganizationSubscription(models.Model):
    STATUS_CHOICES = (
        ('ACTIVE', 'Active'),
        ('EXPIRED', 'Expired'),
        ('CANCELLED', 'Cancelled'),
    )
    BILLING_CHOICES = (
        ('MONTHLY', 'Monthly'),
        ('YEARLY', 'Yearly'),
    )

    org = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscription')
    plan = models.ForeignKey(SubscriptionPlanSetting, on_delete=models.PROTECT, related_name='subscriptions')
    plan_code = models.CharField(max_length=20, blank=True)
    plan_name = models.CharField(max_length=80, blank=True)
    billing_cycle = models.CharField(max_length=10, choices=BILLING_CHOICES, default='MONTHLY')
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='ACTIVE')
    amount_paid = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    reference = models.CharField(max_length=64, unique=True, blank=True)
    subdomain = models.SlugField(max_length=63, blank=True, help_text='Enterprise organization subdomain, for example "acme" for acme.nyscclearance.com')
    starts_at = models.DateTimeField()
    expires_at = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('-updated_at',)

    def __str__(self):
        return f"{self.org.email} - {self.plan_name} ({self.status})"

    def save(self, *args, **kwargs):
        if self.plan:
            self.plan_code = self.plan.code
            self.plan_name = self.plan.name
        if not self.reference:
            self.reference = f"SUB-ADMIN-{uuid.uuid4().hex[:18]}".upper()
        if self.subdomain:
            self.subdomain = self.subdomain.strip().lower()
        super().save(*args, **kwargs)


class SubscriptionPayment(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('SUCCESS', 'Success'),
        ('FAILED', 'Failed'),
    )
    BILLING_CHOICES = OrganizationSubscription.BILLING_CHOICES

    org = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscription_payments')
    plan = models.ForeignKey(SubscriptionPlanSetting, on_delete=models.PROTECT, related_name='payments')
    plan_code = models.CharField(max_length=20)
    plan_name = models.CharField(max_length=80)
    billing_cycle = models.CharField(max_length=10, choices=BILLING_CHOICES, default='MONTHLY')
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    discount_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal('0.00'))
    amount_charged = models.DecimalField(max_digits=12, decimal_places=2)
    reference = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    raw_response = models.JSONField(default=dict, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('-created_at', '-id')

    def __str__(self):
        return f"{self.plan_name} {self.billing_cycle} {self.reference}"


class ClearanceAccess(models.Model):
    SOURCE_CHOICES = (
        ('SUBSCRIPTION', 'Subscription'),
        ('ORG_WALLET', 'Organization Wallet'),
        ('BRANCH_WALLET', 'Admin Wallet'),
        ('CORPER_WALLET', 'Corper Wallet'),
        ('EXISTING_WALLET_CHARGE', 'Existing Wallet Charge'),
    )

    corper = models.ForeignKey('accounts.CorpMember', on_delete=models.CASCADE, related_name='clearance_accesses')
    org = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='clearance_accesses')
    branch = models.ForeignKey(BranchOffice, on_delete=models.SET_NULL, null=True, blank=True, related_name='clearance_accesses')
    reference = models.CharField(max_length=64, unique=True)
    year_month = models.CharField(max_length=6, db_index=True)
    source = models.CharField(max_length=24, choices=SOURCE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created_at', '-id')
        unique_together = ('corper', 'year_month')
        verbose_name = 'Clearance Access'
        verbose_name_plural = 'Clearance Accesses'

    def __str__(self):
        return f"{self.corper.state_code} {self.year_month} via {self.source}"


class ClearanceOverride(models.Model):
    corper = models.ForeignKey('accounts.CorpMember', on_delete=models.CASCADE, related_name='clearance_overrides')
    year_month = models.CharField(max_length=6)  # YYYYMM
    reason = models.CharField(max_length=255, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='granted_clearance_overrides')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('corper', 'year_month')
        ordering = ('-created_at',)
        verbose_name = 'Clearance Override'
        verbose_name_plural = 'Clearance Overrides'

    def __str__(self):
        return f"Override {self.corper.full_name} {self.year_month}"
