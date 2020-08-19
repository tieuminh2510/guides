# Setup Cygwin in Windows XP/2003

 Posted on 2017-06-04 | In [Tool & Skill ](https://morganwu277.github.io/categories/Tool-Skill/), [Windows ](https://morganwu277.github.io/categories/Tool-Skill/Windows/)| [0 Comments](https://morganwu277.github.io/2017/06/04/Setup-Cygwin-in-Windows-XP-2003/#comments)

Everyone should know how to setup [`Cygwin`](https://cygwin.com/) under Windows 7/10, however, if you do it in Windows XP/2003, it will failed.
Why? Because the latest packages are only compatible with Windows 7 or higher.
In other words, the Cygwin respository doesn’t have a mechanism to maintain different version of these binaries. 

So, how to setup Cygwin under Windows XP then? 



The good news is that someone keeps it. 

# 1. Download the setup.exe

32-bit: [setup-x86-2.874.exe](http://ctm.crouchingtigerhiddenfruitbat.org/pub/cygwin/setup/snapshots/setup-x86-2.874.exe)
64-bit: [setup-x86_64-2.874.exe](http://ctm.crouchingtigerhiddenfruitbat.org/pub/cygwin/setup/snapshots/setup-x86_64-2.874.exe)

# 2. Run the setup.exe with -X option

Please open the cmd to run with `-X` option

```
setup-x86-2.874.exe -X
```

# 3. Use the legacy repositories

32-bit: http://ctm.crouchingtigerhiddenfruitbat.org/pub/cygwin/circa/2016/08/30/104223
64-bit: http://ctm.crouchingtigerhiddenfruitbat.org/pub/cygwin/circa/64bit/2016/08/30/104235

Here are the full list of `Cygwin Time Machine Repository`, please use the corresponding respository as the version of `setup.exe`
32-bit: http://ctm.crouchingtigerhiddenfruitbat.org/pub/cygwin/circa/index.html
64-bit: http://ctm.crouchingtigerhiddenfruitbat.org/pub/cygwin/circa/64bit/index.html

Then always use this repository instead of others to install packages. 

Enjoy!!!