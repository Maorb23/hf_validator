from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("infiro_platform", "0001_initial")]
    operations = [
        migrations.AddField(
            model_name="prepared", name="name",
            field=models.CharField(blank=True, default="", max_length=160),
        ),
    ]
