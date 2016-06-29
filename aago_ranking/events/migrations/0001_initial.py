# -*- coding: utf-8 -*-
# Generated by Django 1.9.6 on 2016-06-26 02:10
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import model_utils.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('games', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Event',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', model_utils.fields.AutoCreatedField(default=django.utils.timezone.now, editable=False, verbose_name='created')),
                ('modified', model_utils.fields.AutoLastModifiedField(default=django.utils.timezone.now, editable=False, verbose_name='modified')),
                ('name', models.CharField(max_length=255)),
                ('start_date', models.DateField()),
                ('end_date', models.DateField()),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='EventPlayer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ranking', models.CharField(choices=[('1d', '1dan'), ('2d', '2dan'), ('3d', '3dan'), ('4d', '4dan'), ('5d', '5dan'), ('6d', '6dan'), ('7d', '7dan'), ('8d', '8dan'), ('9d', '9dan'), ('1k', '1kyu'), ('2k', '2kyu'), ('3k', '3kyu'), ('4k', '4kyu'), ('5k', '5kyu'), ('6k', '6kyu'), ('7k', '7kyu'), ('8k', '8kyu'), ('9k', '9kyu'), ('10k', '10kyu'), ('11k', '11kyu'), ('12k', '12kyu'), ('13k', '13kyu'), ('14k', '14kyu'), ('15k', '15kyu'), ('16k', '16kyu'), ('17k', '17kyu'), ('18k', '18kyu'), ('19k', '19kyu'), ('20k', '20kyu'), ('21k', '21kyu'), ('22k', '22kyu'), ('23k', '23kyu'), ('24k', '24kyu'), ('25k', '25kyu'), ('26k', '26kyu'), ('27k', '27kyu'), ('28k', '28kyu'), ('29k', '29kyu'), ('30k', '30kyu')], max_length=4)),
                ('event', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='events.Event')),
                ('player', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='games.Player')),
            ],
        ),
    ]