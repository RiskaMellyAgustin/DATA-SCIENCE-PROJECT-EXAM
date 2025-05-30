# Generated by Django 5.2.1 on 2025-05-21 20:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mellyapp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="AnomalyDetectionLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("gender", models.CharField(max_length=10)),
                ("age", models.IntegerField()),
                ("total_activities_done", models.IntegerField()),
                ("total_duration_minutes", models.FloatField()),
                ("anomaly_score", models.FloatField()),
                ("prediction", models.CharField(max_length=10)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="ClusterAnomalyLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("gender", models.CharField(max_length=10)),
                ("age", models.IntegerField()),
                ("total_activities_done", models.IntegerField()),
                ("total_duration_minutes", models.FloatField()),
                ("cluster_id", models.IntegerField()),
                ("cluster_label", models.CharField(max_length=50)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="DropoutRegressionLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("gender", models.CharField(max_length=10)),
                ("age", models.IntegerField()),
                ("total_activities_done", models.IntegerField()),
                ("total_duration_minutes", models.FloatField()),
                ("dropout_score", models.FloatField()),
                ("prediction", models.CharField(max_length=10)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
