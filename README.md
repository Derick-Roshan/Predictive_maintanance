# Project 1:  Predictive_maintanance 
This is my final year project upon "Motor Predictive Maintanance Failure"
.
.
.
The main concept of this project is Predicting the moter, oil level,  etc.. which is allocated in kind a CNC Machins which is used in Industries.
.
.
Advantage:
- Dont need to wast time for checking each and every compenents if the machin is not working proparly.
- We dont need to invest keep many Engineers to maintain the machins.
- We can know the machins are alright before using.
- Can avoid accidents by machins.

Hosting this Application through AWS cloud platform:

- Developed a Machine Learning application to predict motor failures using sensor data.
- Deployed and hosted the Streamlit application on an AWS EC2 instance running Amazon Linux.
- Configured the server and accessed the application through the EC2 public IP.
- Technologies: Python, Streamlit, Scikit-Learn, AWS EC2, Amazon Linux, Git

Screenshots:

Step 1: Launching an Amazon Linux EC2 instance. image
<img width="628" height="412" alt="image" src="https://github.com/user-attachments/assets/d6535df5-d930-498d-9687-7446fcb6ee5e" />

Step 2: Selecting the Amazon Linux AMI (64-bit). image
<img width="748" height="300" alt="image" src="https://github.com/user-attachments/assets/6086225f-dd18-4ac1-a886-50b1620f6ed5" />

Step 3: Choose an EC2 instance type (t2.medium for this deployment). Becouse the particuler Application need huge performance. image
<img width="820" height="338" alt="image" src="https://github.com/user-attachments/assets/69a050d7-ba3e-4c12-a9d9-ad2926bf154f" />

Step 4: Creating a key pair for secure SSH access to the EC2 instance. image
<img width="578" height="402" alt="image" src="https://github.com/user-attachments/assets/681578c5-8d79-4029-8fd4-a500e47bfc84" />

Step 5: Configure the security group to allow HTTP (port 80) and the application port (e.g. 8502) so the application is accessible from a browser. image
<img width="744" height="354" alt="image" src="https://github.com/user-attachments/assets/b35e9d43-ff53-46a9-ba41-6e1016ae693f" />

Configured the Storage as 30GB. image
<img width="830" height="324" alt="image" src="https://github.com/user-attachments/assets/b6c7f2cf-7713-404f-a5b4-38e7e5115673" />

Number of instance is 1. and clicked on Launch instance. image
<img width="340" height="474" alt="image" src="https://github.com/user-attachments/assets/c3654a84-53dd-4047-a0d8-3334981b5f41" />

You can see that Instance is running. image
<img width="832" height="256" alt="image" src="https://github.com/user-attachments/assets/50b09a3e-197c-42eb-80ac-8b377ec1ddc4" />

I copied a IP address for connecting the Linux server. image
<img width="830" height="190" alt="image" src="https://github.com/user-attachments/assets/98aee0a8-49a3-4fd3-8513-8605f3a50668" />

This is how I connect the Linux server. image
<img width="830" height="388" alt="image" src="https://github.com/user-attachments/assets/09aa367f-e51b-4cdd-a51c-bcafb7a9f134" />

We need to install HTTPD, python3, git, pip3.
We need to clone the project from the github repo. image image
<img width="830" height="398" alt="image" src="https://github.com/user-attachments/assets/c14cb6a4-b998-425e-83f8-25c9ef6f8d7a" />
<img width="830" height="228" alt="image" src="https://github.com/user-attachments/assets/e7a6f9d7-fe60-40cc-bbdb-66a99dd660c7" />

Then we need to download the packages and libraries which we used in the python. image image image image image image
<img width="832" height="104" alt="image" src="https://github.com/user-attachments/assets/b25237ba-7ae0-4cfe-825e-214b29b79f25" />
<img width="830" height="296" alt="image" src="https://github.com/user-attachments/assets/7aee57f1-97dd-4de4-93fb-51b9c7d27d72" />
<img width="830" height="360" alt="image" src="https://github.com/user-attachments/assets/f295ad4d-c382-44d2-98b3-51a8cb08d3bb" />
<img width="830" height="400" alt="image" src="https://github.com/user-attachments/assets/e92fb586-5e92-4fb9-9cd2-9a4a921f5ca1" />
<img width="830" height="360" alt="image" src="https://github.com/user-attachments/assets/2de821af-bbc9-49b7-8526-a8aeaa36c1ff" />
<img width="830" height="84" alt="image" src="https://github.com/user-attachments/assets/90b789f4-93ca-4a04-b9df-22acbeed124a" />

After installing the python packages and if we need to run the project you can see that is running in the bare mode. image
<img width="832" height="358" alt="image" src="https://github.com/user-attachments/assets/12035e4f-80f8-4d03-9f63-37466766e6c6" />

To sort it out: Now you see that the after using nohup command the prject is running in the background. image
<img width="830" height="292" alt="image" src="https://github.com/user-attachments/assets/21df2b6d-26a2-4669-beb6-63a2c561aaae" />

To view the project Iam using the chrom browser. (On incognito tab - Becouse it wont store history or cookies)
http://IP address: 8502
Before that we need to check wheather inbound rules on Security groups has been added - TCP, Port & Source.
<img width="828" height="266" alt="image" src="https://github.com/user-attachments/assets/e37cd0f3-9826-41d3-8466-b36058421931" />

After saving - I entered a IP address and Port number to view the project. image
<img width="830" height="278" alt="image" src="https://github.com/user-attachments/assets/ff5ccede-3a1a-4afc-ace0-033990689d3a" />

Here you go! You can see that the project has been deployed Successfully. image
<img width="830" height="460" alt="image" src="https://github.com/user-attachments/assets/e83a88b1-9d24-405f-aa9a-1babbafc4973" />






