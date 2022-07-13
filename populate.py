import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','test_project.settings')

import django
django.setup()

# using faker

import random
from test_app.models import Topic,webpage,Access_Record
from faker import Faker

fake=Faker()
topics=['search','social','marketplace','news','games']

def add_topics():
    t=Topic.objects.get_or_create(user_name=random.choice(topics))[0]
    t.save()
    return t
def populate(N=5):
    for entry in range(N):
        top=add_topics()

        fake_url=fake.url()
        fake_name=fake.company()
        fake_date=fake.date()


        web_page=webpage.objects.get_or_create(topic=top, name = fake_name, url = fake_url)[0]
        # webpage = object->101

        acc_rec = Access_Record.objects.get_or_create(name = web_page, date = fake_date)[0]

if __name__ == '__main__':
    print("populating script!")
    populate(20)
    print("populating complete!")
