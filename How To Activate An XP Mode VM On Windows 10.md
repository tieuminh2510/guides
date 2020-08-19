# [How To Activate An XP Mode VM On Windows 10](https://www.hardanswers.net/activate-xp-mode-vm-on-windows-10)

Updated about 2 yrs, 5 mths ago (March 8, 2018). Know a better answer? [Let me know](https://www.hardanswers.net/contact)!

# How to activate an XP Mode VM on Windows 10

*You are running Windows 10 and you want keep using an XP Mode virtual machine after its activation has expired.*

Windows XP Mode only works under Windows 7. If you need to run XP on Windows 10, you should start from scratch with a new XP install and a valid XP licence key.

However, if you are one of the many people who have an XP Mode virtual machine and need to continue using it for some reason, this is one way to achieve that.

**Warning: this may violate the XP Mode licensing agreement.**

There are many ways to run an XP VM on Windows 10. One way is to use Oracle VM VirtualBox and either use the existing XP Mode or extract the files from an XP Mode download you have.

However, running XP Mode on anything other than Windows 7 will result in the XP virtual machine not being activated, and there does not appear to be any way to activate it.

If your XP VM has already had its activation expire, you will not be able to login.

To bypass this:

- Start your XP VM in Safe Mode (press F8 while booting)
- Select “Safe Mode with Command Prompt”
- Once it has booted to a command prompt, type "start explorer" to start explorer

You will now be logged into your expired XP VM, but it is still not activated.

To bypass the activation:

- Click on “Start”, then “Run”, and type in “regedit” (presumably “start regedit” in the command prompt would achieve the same thing)
- Go to “HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion\WPAEvents”
  - Expand “WPAEvents”, you’ll see the “OOBETimer”, double-click it, and delete the original value and type in “FF D5 71 D6 8B 6A 8D 6F
     D5 33 93 FD”
  - Click “OK”
- Right click on the “WPAEvents” folder and click on “Permissions…”
  - under “SYSTEM”, click “Deny Full Control”
  - Click “Yes” to the warning message
- Reboot back to normal Windows XP

**Warning: this may violate the XP Mode licensing agreement.**