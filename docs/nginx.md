# NGINX 

NGINX (pronounced "engine-X") is a very fast webserver for serving out static files, or for caching calls to a slower API.

## Why use it?

### File Transfer
Problem: You have a remote server from which you are repeatedly downloading the same files, over and over again, and it is obviously not only wasteful but slow.

Solution: Use a NGINX web server to cache some or all of the files locally so that when you browse to localhost, you get an identical copy of the remote file. 

### Slow API

Problem: A slow API is being called with the same arguments by many different computers, and he exact same value is being recomputed over and over again.

Solution: Use a NGINX web server as a proxy in front of the real API, and let NGINX handle all of the caching. 

## Installation

The first thing to do is turn off any other web servers that are running. In my case, apache2 happened to be installed and running, so I turned that off. 

    sudo service apache2 stop
  
Installing NGINX is straightforward on Ubuntu/Debian:
  
    sudo apt-get install nginx
    sudo service nginx start

In the `/etc/nginx/nginx.conf` file, I put the following in the `http` block. You may want to edit the path `/home/ivar/nginx/cache` to be another directory; that is just where I was testing. To make the cache hold data for 1 week, use this inside the `http` block:

```      
      # 10,080 minutes is 1 week
      proxy_cache_path /home/ivar/nginx_cache levels=1:2 keys_zone=my_cache:10080m max_size=10g
                       inactive=10080m use_temp_path=off;

      server {      	  
         location ~* {
	    # access_log off;
 	    proxy_cache_valid any 10080m;
            proxy_cache my_cache;
            proxy_pass http://potoroo:3003;
         }
      }
```

Finally, force the running server to reload the configuration:

    sudo service nginx restart

You can now navigate to http://localhost/ and observe that it hosts a cached copy.

You can watch the nginx access log in real time using:

    sudo tail -f /var/log/nginx/access.log

## Speed Test

Let's see how this improves the speed. The first call takes about 12 seconds:

```
time wget http://localhost/baphy/271/bbl086b-11-1
```

But the second time you run it, it takes about 52 milliseconds. 

```
time wget http://localhost/baphy/271/bbl086b-11-1
```

## How do I clear the cache?

When you want to clear the cache, simply delete the contents of the nginx_cache directory. 


## Resources

 | For webserver   | http://nginx.org/en/docs/beginners_guide.html                     |
 | For proxy cache | https://www.nginx.com/blog/nginx-caching-guide/                   |
 | For SSL         | http://nginx.org/en/docs/http/configuring_https_servers.html      |
 | For Python      | http://vladikk.com/2013/09/12/serving-flask-with-nginx-on-ubuntu/ |
 | For S3          | https://github.com/anomalizer/ngx_aws_auth                        |
