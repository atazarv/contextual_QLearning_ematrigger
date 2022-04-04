from fabric.api import *

env.user = 'root'
env.hosts = [server_url]


def deploy():
    # creating the distribution
    local('python setup.py sdist')
    # figure out the package name and version
    dist = local('python setup.py --fullname', capture=True).strip()
    filename = '%s.tar.gz' % dist
    # upload the package to the temporary folder on the server
    put('dist/%s' % filename, '/tmp/%s' % filename)
    # install the package in the application's virtualenv with pip
    run('pip3 install --upgrade /tmp/%s' % filename)
    # remove the uploaded package
    run('rm -r /tmp/%s' % filename)
    # touch worker file
    run('touch /var/www/unite/celery_worker.py')
    # restart celery service
    run('service celery restart')
