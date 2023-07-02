import socket
import threading

"""
Port: 
    Communication Links between you and the specific port;
    Anything communicating over a HTTP(8080) port such as
    websites and web-browsers. It is also a host server listening for requests.
    http://localhost/web:localhost(hostname) is the machine name/IP address of the host server.
"""
HEADER = 16 #16 bytes: Max length for a message
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
PORT = 8080 #Common HTTP Port
SERVER = socket.gethostbyname(socket.gethostname()) #Hard coding it would look like: "192.168.1.83"
#print(SERVER) = "192.168.1.83"; Local Device IP Address
#print(socket.gethostname()) = "Yousefs-MBP.fios-router.home"; Device name of local IP Address
print(socket.gethostname())

ADDR = (SERVER,PORT)

"""
    BUILD AND THEN BIND SOCKET WITH ADDRESS:
    First build socket:
    ~Built-in Socket Methods: socket.socket(Socket_Family,Socket_Type)
        ~(Family) = What family type of IP/Address you will be specifying(searching/accepting)
            ~AF_INET(OverTheInternet) = IPv4 Address; AF_INET_6 = IPv6 Address
            
        ~(Type) = Type of way of sending data through a socket
            ~SOCK_STREAM = Tells how to package the data for sending; This type is standard for sending
            data in a sequential order(not random order); Most commonly used for TCP
            ~SOCK_DGRAM = Data is not sent in order.
            Most commonly used for UDP
    
    BIND SOCKET WITH AN ADDRESS: Will allow socket to 'open your device' to other connections:
    Address must be a tuple
    ~Built-in code: "server".bind(Server IP Address,Port in which the server is running off of)
    ~ADDR = (SERVER,PORT)
    ~SERVER = "192.168.1.83"
    ~PORT = 8080
    
    We bounded this socket("server") to this address(ADDR);
    ~Anything that connects to this address will hit this socket
"""
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(ADDR)

def handle_client(conn,addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        #Decode this message from byte(HEADER) to utf-8(FORMAT)
        #Now convert it to integer
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False
            print(f"[{addr}] {msg}")
            conn.send("Message received".encode(FORMAT))
    conn.close()    #Disconnect the current connection

def start():
    server.listen() #Listens for new connections
    print(f"[LISTENING] Server is listening on [{socket.gethostname()}, {SERVER}]")
    while True:
        conn,addr = server.accept()
        """
        conn,addr = server.accept(): Waits for a new connection
        and will store the address of that IP address
        and what port it came from into addr
        
        Then will create an object in conn to allow
        us to send information back to that connection
        """
        thread = threading.Thread(target=handle_client,args=(conn,addr))
        """
            When a new connection occurs, it will first:
            ~Pass that connection to the target(function handle_client)
            ~Give the target arguments: conn and addr from conn,addr = server.accept()
            ~Start the thread
        """
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount()-1}")

print("Server is Starting...")
start()