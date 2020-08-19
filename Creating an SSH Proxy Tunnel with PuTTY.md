# Creating an SSH Proxy Tunnel with PuTTY



<iframe frameborder="0" height="550px" src="https://app.box.com/embed/preview/b4cmudsp4ddd8r00eeqa8d5omvvmtnw1?theme=dark" width="640px" style="box-sizing: border-box;"></iframe>

Some websites available to Math Department members are filtered by the network the traffic originates on. In particular, connections to http://www.ams.org/mathscinet/ must come from a registered UCLA Math IP address to gain full access. If you are browsing this site from off-campus, and you have a Mathnet Linux account, you can use this proxy setup to make it appear that your traffic comes from one of our IP addresses. A proxy setup can be configured using OSX, Linux, or Windows using various browsers. This example shows a connection from a Windows machine using Firefox. Things you'll need: A Linux Mathnet account, PuTTY (ssh client), and Firefox.

 

1. Launch PuTTY and enter the hostname ( <homesite>.math.ucla.edu ) and port. Port should be set to 22.

  - The hostname should be your UCLA homesite followed by ".math.ucla.edu". Login to a linux machine and type "home" and this will display your homesite.

![img](https://www.math.ucla.edu/sites/default/files/computing/kb/images/ssh_tunnel/SSH_Proxy_KB1.jpg)

2. On the left side, in the Category window, go to Connection -> SSH -> Tunnels.

3. For 'Source Port' enter '31415' (this can be configured to whatever you want, just remember it).

4. Under 'Destination' select the 'Dynamic' radio button and leave the 'Auto' button selected.

5. Press the 'Add' button. You should see 'D31415' in the 'Forwarded ports:' box.

6. Then select the 'Open' button. This should open and terminal window and you should be prompted to login.

![img](https://www.math.ucla.edu/sites/default/files/computing/kb/images/ssh_tunnel/SSH_Proxy_KB2.jpg)

 

Once the tunnel is established, you now need to set up a SOCKS proxy in your web browser.

1. Launch Firefox.

2. Go to Tools -> Options.

3. On the left side of the window, select Advanced.

4. Under Advanced, in the middle of the page, select Network -> Connection -> Settings.

![img](https://www.math.ucla.edu/sites/default/files/computing/kb/images/ssh_tunnel/SSH_Proxy_KB3.jpg)

5. Under 'Configure Proxies to Access the Internet' select the 'Manual proxy configuration' radio button.

6. In the 'SOCKS Host' box enter 'localhost' and for 'Port' enter '31415' (or whatever you set your SSH Tunnel up with).

7. Make sure 'SOCKS v5' is selected and select the 'OK' button to save.

![img](https://www.math.ucla.edu/sites/default/files/computing/kb/images/ssh_tunnel/SSH_Proxy_KB4.jpg)

 

As long as your PuTTY SSH connection remains connected, your proxy tunnel will be open and you will be able to use the internet through this proxy.

To determine that the proxy is up:

1. Go to [http://www.whatsmyip.org](http://www.whatsmyip.org/) and confirm that your IP address matches the host IP that you are tunneling through.

2. Also, if you are trying to gain access to MathSciNet, go to http://www.ams.org/mathscinet and look for "Univ of Calif, Los Angeles" in the top right of the page. 

![img](https://www.math.ucla.edu/sites/default/files/computing/kb/images/ssh_tunnel/MathSciNet.jpg)

 