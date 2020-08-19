# Django Rest Framework Permissions example

![img](https://miro.medium.com/max/10368/0*s7nCy3cjz-Ekap7T)

Photo by [Jonathon Young](https://unsplash.com/@jyoung?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral)

In DRF We can use the permissions to implement RBAC (Role-Based Access Control). Role-Based Access Control is an approach that restricts access to users based on their role. You can use Django’s authentication and authorization features to configure Role-Based Access Control.

Django user authentication has built-in models like User, Group, and Permission.

- User: The heart of authentication
- Group: Way of categorizing User
- Permission: Granular access control

This article will not be focussed on the Permission model but will cover some basics on how to write some custom permission for specific groups of users. Group-based permission is one of the different methods for implementing permissions in Django. At the end of this article, I believe, you’ll have a basic concept of custom permission in Django.

Let’s start the example with some project setup

```
$ cd ~/Desktop # your preferred working directory
$ virtualenv -p python3 venv 
$ source venv/bin/activate #activating the virtual environment$ pip install django$ pip install djangorestframework$ django-admin.py startproject django_rest_permission
$ cd django_rest_permission #directory changed inside main project
$ python manage.py startapp user #custom user app
```

All set for the initial project setup, now time to add model for the project

> user.models.py

<iframe src="https://medium.com/media/1ad667f8e23d1edae9346f3451e7859a" allowfullscreen="" frameborder="0" height="527" width="680" title="Django rest framework simple permission example" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 527px;"></iframe>

> We need to do little change in the setting of the project

```
INSTALLED_APPS = [   
..... 
'rest_framework',  #add this line  
'rest_framework.authtoken',  # add this line
'user'  # add this line
].....AUTH_USER_MODEL = 'user.User'  # add this line.....
```

There will be two types of user groups, ***‘admin’\*** and ***‘anonymous\***’. Let’s first add the groups programmatically and then do the migrations. All groups for our project are manipulated using the following ‘***group.py\***’ file. ***‘group.py’\*** file will be placed in the project level directory.

> group.py

<iframe src="https://medium.com/media/91d0b44b618c6a3a7df92c00860d2a1e" allowfullscreen="" frameborder="0" height="373" width="680" title="Default group populate file" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 373px;"></iframe>

> Now time to make migrations and migrate

```
$ python manage.py makemigrations
$ python manage.py migrate
$ python group.py
$ python manage.py createsuperuser  #provide groups.id = 1
```

Time to add serializer for our User model

> user.serializer.py

<iframe src="https://medium.com/media/cf6c98846bf479a36bca47fb24ce87ae" allowfullscreen="" frameborder="0" height="417" width="680" title="Django rest framework permission" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 417px;"></iframe>

Before jumping directly into the views, we will define some permissions in a ‘***permission.py’\*** inside the user app.

> user.permission.py

<iframe src="https://medium.com/media/2152119bf6ac1c0eae7b5dafa5f7246d" allowfullscreen="" frameborder="0" height="1077" width="680" title="Django rest framework permission" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 1077px;"></iframe>

Here we are creating the custom permission class without using any third-party packages.

***has_permission(self, request, view)\*** and ***has_object_permission(self, request, view, obj)\*** methods are two methods defined inside Django’s permission class. ***has_permission()\*** is for the user-level permission while the ***has_object_permission()\*** is object-level permission. These two methods will return boolean (true or false) according to the condition we write inside it. True means the access is allowed otherwise access denied.

In an example of the permission class above, we are checking the group of the user and returning true or false. We can create permission classes as many as our requirements. [Check on official documentation.](https://www.django-rest-framework.org/tutorial/4-authentication-and-permissions/)

Now time to see how these permission class can be associated with a specific view.

> user.views.py

<iframe src="https://medium.com/media/67c75cd2f474a6857e70dbdb936a71d6" allowfullscreen="" frameborder="0" height="967" width="680" title="Django rest permission" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 967px;"></iframe>

Let’s configure URLs to our API endpoint. Remember the following ***urls.py\*** file needs to be inside our user app.

> user.urls.py

<iframe src="https://medium.com/media/b47a90c9b969c55a811549f241502e33" allowfullscreen="" frameborder="0" height="329" width="680" title="Django rest permission user urls" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 329px;"></iframe>

At last, we need to add the project level URL.

> urls.py

<iframe src="https://medium.com/media/bbe53d95dfcde3bf0fef8bead7c4882a" allowfullscreen="" frameborder="0" height="197" width="680" title="Django rest permission project url" class="cg t u fw ak" scrolling="auto" style="box-sizing: inherit; top: 0px; left: 0px; width: 680px; position: absolute; height: 197px;"></iframe>

Generating token is not the main theme of this article but you can post the data as shown in the picture below.

![img](https://miro.medium.com/max/3744/1*GCbt5kkcIOzEK3xmtOKKpw.png)

login endpoint for API generating a token.

You can use the ***‘token’\*** from the above response and add it in the ***header\*** of the HTTP request as shown in the picture below.

![img](https://miro.medium.com/max/3740/1*a8dijEXdsaRZ_BFhPLxO-g.png)

adding a token to the header of the HTTP request

The user in a group can have the following role:

**Admin** group user can

1. Login
2. Add a new user
3. List all users
4. View individual user
5. Delete users
6. Update all users

**Anonymous** group user can

1. Login
2. List all users
3. View and Update own detail

***Without login, no user can get access to the system.\***

There are other ways that we can implement RBAC. I have done this doing some research over the internet. Any review of this approach will be highly appreciated.

You can find the Github Repository link to this example article [here](https://github.com/depeshpo/django_rest_permission).