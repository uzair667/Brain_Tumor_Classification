# Generated by Django 3.2.8 on 2021-11-15 17:58

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('test_app', '0003_auto_20211030_1154'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(max_length=264)),
                ('last_name', models.CharField(max_length=264)),
                ('email', models.EmailField(max_length=264, unique=True)),
                ('password', models.CharField(max_length=128)),
            ],
        ),
        migrations.RemoveField(
            model_name='webpage',
            name='topic',
        ),
        migrations.DeleteModel(
            name='Topic',
        ),
        migrations.AlterField(
            model_name='access_record',
            name='name',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='test_app.user'),
        ),
        migrations.DeleteModel(
            name='webpage',
        ),
    ]