# Generated by Django 5.1.5 on 2025-02-01 23:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0004_remove_book_published_date'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='book',
            name='title',
        ),
    ]
