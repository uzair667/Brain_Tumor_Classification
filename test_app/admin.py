from django.contrib import admin
# from test_app.models import users,website,records
# # Register your models here.
# admin.site.register(users)
# admin.site.register(website)
# admin.site.register(records)


from test_app.models import Topic,webpage,Access_Record

admin.site.register(Topic)
admin.site.register(webpage)
admin.site.register(Access_Record)
