from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("infiro_platform", "0002_prepared_name")]
    operations = [migrations.AddField(
        model_name="recommendation", name="name",
        field=models.CharField(blank=True, default="", max_length=160),
    )]
