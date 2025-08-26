from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models
from django.conf import settings


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
    number_of_corpers = models.PositiveIntegerField(default=0)

    is_active = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    is_email_verified = models.BooleanField(default=False)
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
    logo = models.ImageField(upload_to='org_logos/', blank=True, null=True)
    late_time = models.TimeField(blank=True, null=True)
    closing_time = models.TimeField(blank=True, null=True)
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


class Department(models.Model):
    branch = models.ForeignKey(BranchOffice, on_delete=models.CASCADE, related_name='departments')
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name} - {self.branch.name}"


class Unit(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='units')
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.name} - {self.department.name}"


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
    account = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='corper_profile')
    branch = models.ForeignKey(BranchOffice, on_delete=models.SET_NULL, null=True, blank=True, related_name='corpers')
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True, related_name='corpers')
    unit = models.ForeignKey(Unit, on_delete=models.SET_NULL, null=True, blank=True, related_name='corpers')
    face_encoding = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.full_name} ({self.state_code})"
