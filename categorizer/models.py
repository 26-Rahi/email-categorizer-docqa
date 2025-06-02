from django.db import models

class EmailHistory(models.Model):
    content = models.TextField()
    category = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return f"{self.category} - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"