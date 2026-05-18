from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from pathlib import Path
from rest_framework import serializers
from django.conf import settings

from .tokens import generate_email_token
from .models import OrganizationProfile, BranchOffice, Department, Unit, CorpMember, PublicHoliday, LeaveRequest, Notification, WalletAccount, WalletTransaction
from .models import QueryRecord
from django.core.validators import RegexValidator


User = get_user_model()


class OrganizationRegisterSerializer(serializers.ModelSerializer):
    # Password is optional: org users can set it after email verification
    # (similar to invited branch/corper flows).
    password = serializers.CharField(write_only=True, min_length=8, required=False, allow_blank=True)
    password_confirm = serializers.CharField(write_only=True, min_length=8, required=False, allow_blank=True)
    location_lat = serializers.FloatField(required=False, allow_null=True)
    location_lng = serializers.FloatField(required=False, allow_null=True)

    class Meta:
        model = User
        fields = (
            'email', 'name', 'address', 'phone_number',
            'password', 'password_confirm', 'location_lat', 'location_lng'
        )

    def validate(self, attrs):
        password = attrs.get('password')
        password_confirm = attrs.get('password_confirm')
        # Only validate mismatch when a password was provided.
        if password or password_confirm:
            if password != password_confirm:
                raise serializers.ValidationError({'password_confirm': 'Passwords do not match.'})
        return attrs

    def create(self, validated_data):
        password = (validated_data.pop('password', '') or '').strip()
        validated_data.pop('password_confirm', None)
        loc_lat = validated_data.pop('location_lat', None)
        loc_lng = validated_data.pop('location_lng', None)
        user = User.objects.create_user(**validated_data)

        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()

        user.is_active = False
        user.is_email_verified = False
        user.save()

        token = generate_email_token(user.id)
        verify_url = self._build_verify_url(token)
        self._send_verification_email(user.email, user.name, verify_url)

        # Create initial profile with optional coordinates
        OrganizationProfile.objects.get_or_create(
            user=user,
            defaults={'location_lat': loc_lat, 'location_lng': loc_lng}
        )

        return user

    def _build_verify_url(self, token: str) -> str:
        # Prefer frontend URL so users click directly into the SPA
        request = self.context.get('request')
        try:
            request_origin = request.headers.get('Origin') if request else None
            allowed = set(getattr(settings, 'FRONTEND_ORIGINS', []))
        except Exception:
            request_origin = None
            allowed = set()
        base = (request_origin if request_origin in allowed else getattr(settings, 'FRONTEND_ORIGIN', 'http://localhost:5173')).rstrip('/')
        return f"{base}/verify-success?token={token}"

    def _send_verification_email(self, email: str, name: str, url: str) -> None:
        subject = 'Verify your NYSC Clearance account'
        message = (
            f"Hello {name},\n\n"
            f"Please verify your email to activate your account.\n"
            f"Click the link below (valid for 24 hours):\n{url}\n\n"
            f"If you didn't request this, you can ignore this email."
        )
        send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    role = serializers.ChoiceField(choices=[('ORG','ORG'),('BRANCH','BRANCH'),('CORPER','CORPER')], required=False)


class OrganizationProfileSerializer(serializers.ModelSerializer):
    logo = serializers.FileField(required=False, allow_null=True)
    nullable_blank_fields = {
        'late_time',
        'closing_time',
        'max_days_late',
        'max_days_absent',
        'location_lat',
        'location_lng',
    }
    optional_file_fields = {'logo', 'signature'}
    allowed_logo_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.webp', '.gif', '.bmp'}

    class Meta:
        model = OrganizationProfile
        fields = (
            'late_time', 'closing_time', 'max_days_late', 'max_days_absent',
            'logo', 'location_lat', 'location_lng',
            'signatory_name', 'signature'
        )

    def to_internal_value(self, data):
        if hasattr(data, 'copy'):
            data = data.copy()
        for field in self.nullable_blank_fields:
            if field in data and data.get(field) == '':
                data[field] = None
        for field in self.optional_file_fields:
            if field not in data:
                continue
            value = data.get(field)
            is_empty_upload = getattr(value, 'size', None) == 0 and not getattr(value, 'name', '')
            if value in ('', None) or is_empty_upload:
                try:
                    data.pop(field)
                except Exception:
                    data[field] = None
        return super().to_internal_value(data)

    def validate_logo(self, value):
        if not value:
            return value
        ext = Path(getattr(value, 'name', '') or '').suffix.lower()
        if ext not in self.allowed_logo_extensions:
            allowed = ', '.join(sorted(self.allowed_logo_extensions))
            raise serializers.ValidationError(f'Unsupported logo format. Please upload one of: {allowed}.')
        return value

    def update(self, instance, validated_data):
        # Do not overwrite existing logo when no new file is provided
        if 'logo' in validated_data:
            logo = validated_data.get('logo')
            drop = False
            if logo is None or logo == '':
                drop = True
            else:
                size = getattr(logo, 'size', None)
                if size == 0:
                    drop = True
            if drop:
                validated_data.pop('logo', None)
        # Do not overwrite existing signature when no new file is provided
        if 'signature' in validated_data:
            signature = validated_data.get('signature')
            drop_sig = False
            if signature is None or signature == '':
                drop_sig = True
            else:
                size = getattr(signature, 'size', None)
                if size == 0:
                    drop_sig = True
            if drop_sig:
                validated_data.pop('signature', None)
        return super().update(instance, validated_data)


class BranchOfficeSerializer(serializers.ModelSerializer):
    # Incoming-only admin invite fields
    admin_name = serializers.CharField(write_only=True, required=False, allow_blank=True)
    admin_email = serializers.EmailField(write_only=True, required=False, allow_blank=True)
    admin_staff_id = serializers.CharField(write_only=True, required=False, allow_blank=True)
    # Outgoing admin info
    admin_info = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = BranchOffice
        fields = (
            'id', 'name', 'address', 'latitude', 'longitude',
            'admin_name', 'admin_email', 'admin_staff_id',
            'admin_info'
        )

    def validate(self, attrs):
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        name = (attrs.get('name') or getattr(self.instance, 'name', '') or '').strip()
        if name and getattr(user, 'role', None) == 'ORG':
            qs = BranchOffice.objects.filter(user=user, name__iexact=name)
            if self.instance:
                qs = qs.exclude(pk=self.instance.pk)
            if qs.exists():
                raise serializers.ValidationError({'name': 'A branch with this name already exists.'})
        return attrs

    def create(self, validated_data):
        admin_name = validated_data.pop('admin_name', '').strip()
        admin_email = validated_data.pop('admin_email', '').strip()
        admin_staff_id = validated_data.pop('admin_staff_id', '').strip()
        user = self.context['request'].user

        branch = BranchOffice.objects.create(user=user, **validated_data)

        # If admin details provided, create invited user and link
        if admin_email:
            # Avoid duplicate accounts for the same email
            admin_user, created = User.objects.get_or_create(
                email=admin_email,
                defaults={
                    'name': admin_name or admin_email.split('@')[0],
                    'is_active': False,
                    'is_email_verified': False,
                    'role': 'BRANCH',
                }
            )
            if created:
                admin_user.set_unusable_password()
                admin_user.save()
            branch.admin = admin_user
            branch.admin_staff_id = admin_staff_id
            branch.save(update_fields=['admin', 'admin_staff_id'])

            token = generate_email_token(admin_user.id)
            verify_url = self._build_verify_url(token)
            self._send_verification_email(admin_user.email, admin_user.name, verify_url)

        return branch

    def _build_verify_url(self, token: str) -> str:
        base = self.context.get('request').build_absolute_uri('/api/auth/verify/')
        return f"{base}?token={token}&role=BRANCH"

    def _send_verification_email(self, email: str, name: str, url: str) -> None:
        subject = 'You are invited as Branch Admin'
        message = (
            f"Hello {name},\n\n"
            f"You have been invited as a Branch Admin on NYSC Clearance.\n"
            f"Please verify your email to activate your access and set your password:\n{url}\n\n"
            f"If you didn't expect this, you can ignore this email."
        )
        send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])

    def get_admin_info(self, obj):
        if not obj.admin:
            return None
        return {
            'name': getattr(obj.admin, 'name', ''),
            'email': getattr(obj.admin, 'email', ''),
            'staff_id': obj.admin_staff_id,
        }

    def update(self, instance, validated_data):
        admin_name = validated_data.pop('admin_name', None)
        admin_email = validated_data.pop('admin_email', None)
        admin_staff_id = validated_data.pop('admin_staff_id', None)

        # Apply basic field updates first
        for attr, val in validated_data.items():
            setattr(instance, attr, val)

        # Track previous admin email to decide if we should send a fresh invite
        prev_admin_email = (getattr(instance.admin, 'email', None) or '').lower()
        invite_needed = False

        if admin_email is not None:
            admin_email = admin_email.strip()
            if admin_email:
                if admin_email.lower() != prev_admin_email:
                    # New/different admin email → (re)assign and send activation link
                    admin_user, created = User.objects.get_or_create(
                        email=admin_email,
                        defaults={
                            'name': (admin_name or admin_email.split('@')[0]) if admin_name is not None else admin_email.split('@')[0],
                            'is_active': False,
                            'is_email_verified': False,
                            'role': 'BRANCH',
                        }
                    )
                    if created:
                        admin_user.set_unusable_password()
                        admin_user.save()
                    instance.admin = admin_user
                    invite_needed = True
                else:
                    # Same admin email; optionally update name if provided
                    if admin_name is not None and instance.admin:
                        new_name = admin_name.strip()
                        if new_name and new_name != instance.admin.name:
                            instance.admin.name = new_name
                            instance.admin.save(update_fields=['name'])
            else:
                # Empty string clears admin
                instance.admin = None

        if admin_staff_id is not None:
            instance.admin_staff_id = admin_staff_id

        instance.save()

        if invite_needed and instance.admin:
            token = generate_email_token(instance.admin.id)
            verify_url = self._build_verify_url(token)
            self._send_verification_email(instance.admin.email, instance.admin.name, verify_url)

        return instance


class DepartmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Department
        fields = ('id', 'name')

    def validate(self, attrs):
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        name = (attrs.get('name') or getattr(self.instance, 'name', '') or '').strip()
        if not name:
            raise serializers.ValidationError({'name': 'Department name is required.'})
        # Enforce unique department name per organisation (case-insensitive)
        org_user = user
        if getattr(user, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            org_user = b.user if b else None
        if org_user:
            qs = Department.objects.filter(user=org_user, name__iexact=name)
            if self.instance:
                qs = qs.exclude(pk=self.instance.pk)
            if self.instance and qs.exists():
                raise serializers.ValidationError({'name': 'A department with this name already exists in your organisation.'})
        return attrs

    def create(self, validated_data):
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        name = (validated_data.get('name') or '').strip()

        org_user = user
        if getattr(user, 'role', None) == 'ORG':
            org_user = user
        elif getattr(user, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            org_user = b.user if b else None
        else:
            raise serializers.ValidationError('Not allowed')

        dept = Department.objects.filter(user=org_user, name__iexact=name).first()
        if dept:
            return dept
        return Department.objects.create(user=org_user, name=name)


class UnitSerializer(serializers.ModelSerializer):
    class Meta:
        model = Unit
        fields = ('id', 'name')

    def validate(self, attrs):
        name = (attrs.get('name') or getattr(self.instance, 'name', '') or '').strip()
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        org_user = user
        if getattr(user, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            org_user = b.user if b else None
        if org_user and name:
            qs = Unit.objects.filter(user=org_user, name__iexact=name)
            if self.instance:
                qs = qs.exclude(pk=self.instance.pk)
            if qs.exists():
                raise serializers.ValidationError({'name': 'A unit with this name already exists in your organisation.'})
        return attrs

    def create(self, validated_data):
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        org_user = user
        if getattr(user, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=user).first()
            org_user = b.user if b else None
        if not org_user:
            raise serializers.ValidationError('Not allowed')
        name = (validated_data.get('name') or '').strip()
        unit = Unit.objects.filter(user=org_user, name__iexact=name).first()
        if unit:
            return unit
        return Unit.objects.create(user=org_user, name=name)


class PublicHolidaySerializer(serializers.ModelSerializer):
    class Meta:
        model = PublicHoliday
        fields = ('id', 'title', 'start_date', 'end_date')


class LeaveRequestSerializer(serializers.ModelSerializer):
    corper_name = serializers.CharField(source='corper.full_name', read_only=True)
    class Meta:
        model = LeaveRequest
        fields = ('id', 'corper', 'corper_name', 'branch', 'start_date', 'end_date', 'reason', 'status')
        read_only_fields = ('status', 'corper', 'branch')

    def validate(self, attrs):
        start = attrs.get('start_date')
        end = attrs.get('end_date')
        if start and end and end < start:
            raise serializers.ValidationError({'end_date': 'End date cannot be earlier than start date'})
        return attrs


state_code_validator = RegexValidator(
    regex=r'^[A-Z]{2}/\d{2}[A-Z]/\d{4}$',
    message='StateCode must be in format AA/00A/0000'
)


class CorpMemberSerializer(serializers.ModelSerializer):
    state_code = serializers.CharField(validators=[state_code_validator])
    email = serializers.EmailField()
    cds_day = serializers.IntegerField(required=False, allow_null=True)
    organization_id = serializers.IntegerField(source='user_id', read_only=True)

    class Meta:
        model = CorpMember
        fields = (
            'id', 'full_name', 'email', 'gender', 'state_code', 'passing_out_date',
            'branch', 'department', 'unit', 'cds_day', 'organization_id', 'face_encoding'
        )
        read_only_fields = ('face_encoding',)

    def validate(self, attrs):
        # For organization users, branch is required. For branch admins, default later.
        request = self.context.get('request')
        user = getattr(request, 'user', None)
        # Only enforce branch on create; on update we can infer from instance
        if self.instance is None and getattr(user, 'role', None) == 'ORG' and not attrs.get('branch'):
            raise serializers.ValidationError({'branch': 'Branch is required'})

        cds = attrs.get('cds_day')
        if cds is not None:
            try:
                cds_int = int(cds)
            except Exception:
                raise serializers.ValidationError({'cds_day': 'Invalid CDS day'})
            if cds_int < 0 or cds_int > 4:
                raise serializers.ValidationError({'cds_day': 'CDS day must be Monday–Friday'})
            attrs['cds_day'] = cds_int
        return attrs

    def create(self, validated_data):
        # Create or reuse a user account for the corper and invite
        email = validated_data.get('email')
        full_name = validated_data.get('full_name')
        request = self.context.get('request')
        owner = request.user if request else None

        # Do not allow enrolling with an email that already exists.
        # This avoids role conflicts and makes onboarding explicit.
        if email and User.objects.filter(email=email).exists():
            raise serializers.ValidationError({'email': 'A user with this email already exists.'})

        corper_user = User.objects.create(
            email=email,
            name=full_name,
            is_active=False,
            is_email_verified=False,
            role='CORPER',
        )
        corper_user.set_unusable_password()
        corper_user.save()

        # Determine owning organization and default branch if created by a branch admin
        from .models import BranchOffice
        org_user = owner
        if getattr(owner, 'role', None) == 'BRANCH':
            b = BranchOffice.objects.filter(admin=owner).first()
            if b:
                org_user = b.user
                # Default branch to admin's branch if not provided
                if not validated_data.get('branch'):
                    validated_data['branch'] = b
        elif getattr(owner, 'role', None) == 'CORPER':
            raise serializers.ValidationError('Not allowed')

        # Department and unit are organisation-wide (not tied to offices and not nested).
        dept = validated_data.get('department')
        unit = validated_data.get('unit')
        if dept and getattr(dept, 'user_id', None) != getattr(org_user, 'id', None):
            raise serializers.ValidationError({'department': 'Invalid department for this organisation'})
        if unit and getattr(unit, 'user_id', None) != getattr(org_user, 'id', None):
            raise serializers.ValidationError({'unit': 'Invalid unit for this organisation'})

        cm = CorpMember.objects.create(user=org_user, account=corper_user, **validated_data)

        token = generate_email_token(corper_user.id)
        verify_url = self._build_verify_url(token)
        self._send_verification_email(corper_user.email, corper_user.name, verify_url)

        return cm

    def _build_verify_url(self, token: str) -> str:
        base = self.context.get('request').build_absolute_uri('/api/auth/verify/')
        return f"{base}?token={token}&role=CORPER"

    def _send_verification_email(self, email: str, name: str, url: str) -> None:
        subject = 'Complete your NYSC Clearance account'
        message = (
            f"Hello {name},\n\n"
            f"You have been enrolled as a corper.\n"
            f"Please verify your email to activate your account and set your password:\n{url}\n\n"
            f"If you didn't expect this, you can ignore this email."
        )
        send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email])

    def update(self, instance, validated_data):
        # When updating, ensure department/unit belong to organisation.
        dept = validated_data.get('department', instance.department)
        unit = validated_data.get('unit', instance.unit)
        if dept and getattr(dept, 'user_id', None) != getattr(instance.user, 'id', None):
            raise serializers.ValidationError({'department': 'Invalid department for this organisation'})
        if unit and getattr(unit, 'user_id', None) != getattr(instance.user, 'id', None):
            raise serializers.ValidationError({'unit': 'Invalid unit for this organisation'})

        # Handle email change: update the corper login account and force re-activation.
        new_email = validated_data.get('email')
        if new_email is not None:
            new_email = (new_email or '').strip().lower()
            if not new_email:
                raise serializers.ValidationError({'email': 'Email is required'})

            account = getattr(instance, 'account', None)
            if not account:
                raise serializers.ValidationError({'email': 'Corper account not found'})

            current_email = (getattr(account, 'email', '') or '').lower()
            if new_email != current_email:
                # Avoid conflicts with existing users
                if User.objects.filter(email=new_email).exclude(id=account.id).exists():
                    raise serializers.ValidationError({'email': 'A user with this email already exists.'})

                account.email = new_email
                account.is_active = False
                account.is_email_verified = False
                account.set_unusable_password()
                account.save(update_fields=['email', 'is_active', 'is_email_verified', 'password'])

                # Keep CorpMember.email in sync
                instance.email = new_email

                token = generate_email_token(account.id)
                verify_url = self._build_verify_url(token)
                self._send_verification_email(account.email, account.name, verify_url)

                # Prevent DRF from trying to set instance.email again
                validated_data.pop('email', None)

        # Keep user's display name aligned to full_name (optional)
        if 'full_name' in validated_data and instance.account:
            try:
                instance.account.name = validated_data.get('full_name')
                instance.account.save(update_fields=['name'])
            except Exception:
                pass

        return super().update(instance, validated_data)


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ('id', 'title', 'message', 'branch', 'created_at')


class QueryRecordSerializer(serializers.ModelSerializer):
    corper_name = serializers.CharField(source='corper.full_name', read_only=True)
    corper_state_code = serializers.CharField(source='corper.state_code', read_only=True)
    branch_name = serializers.CharField(source='branch.name', read_only=True)
    created_by_email = serializers.CharField(source='created_by.email', read_only=True)
    replied_by_email = serializers.CharField(source='replied_by.email', read_only=True)

    class Meta:
        model = QueryRecord
        fields = (
            'id',
            'org', 'branch', 'branch_name',
            'corper', 'corper_name', 'corper_state_code',
            'title', 'message', 'status',
            'corper_reply', 'replied_at', 'replied_by', 'replied_by_email',
            'created_by', 'created_by_email', 'resolved_by',
            'created_at', 'updated_at',
        )
        read_only_fields = (
            'org',
            'created_by',
            'resolved_by',
            'created_at',
            'updated_at',
            'corper_reply',
            'replied_at',
            'replied_by',
        )


class WalletTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WalletTransaction
        fields = ('id', 'type', 'amount', 'vat_amount', 'total_amount', 'description', 'reference', 'created_at')


class WalletAccountSerializer(serializers.ModelSerializer):
    transactions = WalletTransactionSerializer(many=True, read_only=True)

    class Meta:
        model = WalletAccount
        fields = ('balance', 'transactions')
