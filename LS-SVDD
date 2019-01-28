##import dataset

charlie=pd.read_csv("D:\statistic classes\statistical computing\homework\charlie.csv",sep=',')

charlie=pd.DataFrame(data=charlie)
target={'Original':1,'New':-1}
charlie['Data']=charlie['Data'].map(target)
print(charlie)
##new features z1 and z2 and single value reponse orginal 
Z,original=charlie.ix[0:19,6:8].values,charlie.ix[0:19,0].values
Z_test,new_test=charlie.ix[20:30,6:8].values,charlie.ix[20:30,0].values

#compute the K matrix 

def kernel(x,y,sigma):
    K=np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            delta=x[i]-y[j]
            sumsquare=float(delta.dot(delta.T))
            K[i,j]=np.exp(-0.5*sumsquare/(sigma**2))
    return K
sigma=5**0.5
K=kernel(Z,Z,sigma)
#K is a 20*20 matrix
lk=np.ones((len(Z),1))   #lk denotes lower kj=k(xj,xj)
## compute alpha by above formula
import scipy.special,scipy.linalg
def alpha(C,length,K):
    Ematr=np.ones((length,1))
    ET=Ematr.T
    Hmatr=K+0.5*np.eye(length)/C
    HI=np.linalg.inv(Hmatr)
    Sol1=(2-np.dot(np.dot(ET,HI),lk))/(np.dot(np.dot(ET,HI),Ematr))
    Sol2=lk+Sol1*Ematr
    alpha=0.5*np.dot(HI,Sol2)
    return alpha
## apply to dataset charlie and obtained alpha
alf=alpha(8,20,K)   
## problem 2
##compute the radius Rsquare   function
def KF3(a,x,y,sigma):            # the third part sum(alpha*alpha*K)
    kf3=np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            kf3[i,j] +=a[j]*a[i]*kernel(x,y,sigma)[i,j]
    return kf3
KF33=sum(sum(KF3(alf,Z,Z,sigma))) 
def KF2(a,x,y,sigma):            # the second part sum(alpha*alpha*K)
    kf2=np.zeros(20)
    for i in range(20):
        for j in range(20):
            kf2[i] +=-2*a[j]*kernel(x,y,sigma)[i,j]
    return kf2 
KF22=np.mean(KF2(alf,Z,Z,sigma))
def R_square(x,y):
    R_square=x+y+1
    return R_square
R2=R_square(KF33,KF22)

## compute the distance D
def distance(x,y,a,sigma):
    dis=np.zeros(len(x))
    for i in range(len(x)):
        dis[i]=1+KF33
        for j in range(len(y)):
            dis[i] +=-2*a[j]*kernel(x,y,sigma)[i,j]
    return dis
D=distance(Z_test,Z,alf,sigma)       
def err(x,y):
    count=0
    for i in range(len(x)):
        if x[i]<y:
            count +=1
    err=count/len(x)
    return err
print('The error rates is %s when sigma=5^(0.5) and c=8' %err(D,R2))

## find boundary

#calculate the center of the ball
def center(a,x):
    center=np.dot(a.T,x)
    return center
Cen1=center(alf,Z)
## drawing the boundary
ztest1=Z_test[:,0]
ztest2=Z_test[:,1]
def boun(R,Ce):
    plt.plot(Z[:,0],Z[:,1],'bs')
    plt.plot(ztest1,ztest2,'ro')
    t=np.arange(0,2*np.pi,0.1)
    x=Ce[0,0]+np.sqrt(R)*np.sin(t)
    y=Ce[0,1]+np.sqrt(R)*np.cos(t)
    plt.plot(x,y)
    plt.title("R^2=%s" %R)
    plt.show()
##trying several R_square
K2=kernel(Z,Z,0.001)  #sigma=0.5
K3=kernel(Z,Z,1)  #sigma=5 
K4=kernel(Z,Z,5) #sigma=10
alf2=alpha(15,20,K2)  #c=8
alf3=alpha(10,20,K3)  #c=5
alf4=alpha(5,20,K4) #c=10
R22=R_square(sum(sum(KF3(alf2,Z,Z,0.001))),np.mean(KF2(alf2,Z,Z,0.001)))
R23=R_square(sum(sum(KF3(alf3,Z,Z,1))),np.mean(KF2(alf3,Z,Z,1)))
R24=R_square(sum(sum(KF3(alf4,Z,Z,5))),np.mean(KF2(alf4,Z,Z,5)))
Cen2=center(alf2,Z)
Cen3=center(alf3,Z)
Cen4=center(alf4,Z)
## draw boundary

boun(R2,Cen1)
boun(R22,Cen2)
boun(R23,Cen3)
boun(R24,Cen4)



###another method to draw boundary by find classifier=R^2-norm(x-center)
##compute classifier
zvar1=Z[:,0]
zvar2=Z[:,1]
ztest1=Z_test[:,0]
ztest2=Z_test[:,1]
## find boundary
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#calculate the center of the ball
def center(a,x):
    center=np.dot(a.T,x)
    return center
Cen1=center(alf,Z)


##trying several R_square
K2=kernel(Z,Z,0.05)  #sigma=0.5
K3=kernel(Z,Z,5)  #sigma=5 
K4=kernel(Z,Z,2) #sigma=10
alf2=alpha(10,20,K2)  #c=8
alf3=alpha(5,20,K3)  #c=5
alf4=alpha(10,20,K4) #c=10
R22=R_square(sum(sum(KF3(alf2,Z,Z,0.05))),np.mean(KF2(alf2,Z,Z,0.05)))
R23=R_square(sum(sum(KF3(alf3,Z,Z,5))),np.mean(KF2(alf3,Z,Z,5)))
R24=R_square(sum(sum(KF3(alf4,Z,Z,2))),np.mean(KF2(alf4,Z,Z,2)))
Cen2=center(alf2,Z)
Cen3=center(alf3,Z)
Cen4=center(alf4,Z)


##boundary R^2-norm(x-center)
def classifier(rs,x,y,cen):
    classi=[]
    for i in range(20):
        classi=rs-(x-cen[0,0])**2-(y-cen[0,1])**2
    return classi
x1=np.reshape(np.arange(-2,4,0.1),60,1)
y1=np.reshape(np.arange(-2,4,0.1),60,1)

clas1=classifier(R2,zvar1,zvar2,Cen1)
clas2=classifier(R22,zvar1,zvar2,Cen2)
clas3=classifier(R23,zvar1,zvar2,Cen3)
clas4=classifier(R24,zvar1,zvar2,Cen4)
##import dataset find min and max of two features in order to plot meshgrid
x_min, x_max=zvar1.min() - 1, zvar1.max() + 1 ## feature1 with min -3.656 and max 4.2
y_min, y_max=zvar2.min() - 1, zvar2.max() + 1 ##feature2 with min -3.29 and max 2.67
x1=np.arange(x_min,x_max,0.5)
y1=np.arange(y_min,x_max,0.5)
X,Y=np.meshgrid(x1,y1)

def boundary(cla):
    fig,ax=plt.subplots()
    ax.tricontour(zvar1.ravel(),zvar2.ravel(),cla.ravel(),levels=[0,1],colors='k',linestyles=['--','--','--'])
    ax.scatter(zvar1,zvar2,marker='s',edgecolors='k')       
    ax.scatter(ztest1,ztest2,marker='o',edgecolors='k')    
    ax.set_ylim(-4,4)
    ax.set_xlim(-4,7)
    ax.set_xlabel('feature z1')
    ax.set_ylabel('feature z2')
    ax.set_title('kernel classifier by method SVDD')   
    return plt.show()

boundary(clas1)
boundary(clas2)
boundary(clas3)
boundary(clas4)
