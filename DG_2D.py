#-------------------------------------------------------------------#
#This code solves the 2D Gaussian wave equation for a given order(n)
#--------------------------------------------------and elementa(Nel)
#wriiten by Sivakesava Venum for the fulfilment of M.Tech degree
#           Mechanical Engineering Department, IIT BOMBAY
#---------------------------------------------------------------------#
import numpy
from numpy import matrix,array
from numpy import linalg
import math
#---------------------------------------------------------------------#
############################################################################
#This code computes the Legendre Polynomials and its 1st and 2nd derivatives
############################################################################
#---------------------------------------------------------------------#
def legendre_poly(p,x):
   "function"
   L1,L1_1,L1_2=0,0,0
   L0,L0_1,L0_2=1,0,0

   for i in range(1,p+1):
       L2,L2_1,L2_2=L1,L1_1,L1_2
       L1,L1_1,L1_2=L0,L0_1,L0_2
       a=(2.0*i-1.0)/i
       b=(i-1.0)/i
       L0=a*x*L1 - b*L2
       L0_1=a*(L1 + x*L1_1) - b*L2_1
       L0_2=a*(2*L1_1 + x*L1_2) - b*L2_2
   return L0,L0_1,L0_2
#---------------------------------------------------------------------#
####################################################################
#This code computes the Legendre-Gauss-Lobatto points and weights
#which are the roots of the Lobatto Polynomials.
####################################################################
#---------------------------------------------------------------------#
def legendre_gauss_lobatto(P):
    "function"
    import math
    p=P-1 #Order of the Polynomials
    ph=int( (p+1)/2 )
    from numpy import matrix
    xgl=[0 for i in range(p+2)]
    wgl=[0 for i in range(p+2)]
    for i in range(1,ph+1):
        x=math.cos( (2*i-1)*math.pi/(2*p+1) )
        for k in range(1,21):
            [L0,L0_1,L0_2]=legendre_poly(p,x) #Compute Nth order Derivatives of Legendre Polys
            #Get Nelw Nelwton Iteration
            dx=-(1-x**2)*L0_1/(-2*x*L0_1 + (1-x**2)*L0_2)
            x=x+dx
            if (abs(dx) < 10**-20):
                break
        xgl[p+2-i]=x
        wgl[p+2-i]=2/(p*(p+1)*L0**2)
       
    #Check for Zero Root
    if (p+1 != 2*ph):
        x=0
        [L0,L0_1,L0_2]=legendre_poly(p,x)
        xgl[ph+1]=x
        wgl[ph+1]=2/(p*(p+1)*L0**2)
   
#Find remainder of roots via symmetry
    for i in range(1,ph+1):
        xgl[i]=-xgl[p+2-i]
        wgl[i]=+wgl[p+2-i]
    del xgl[0]
    del wgl[0]
    return xgl,wgl
#-----------------------------------------------------------------#
###################################################################
#The following function calculate the mass,differentiation and
#flux matrices using either exact or inexact integration
###################################################################
#-----------------------------------------------------------------#
def mass_diff_flux(zcal,wcal,z,w,N,Nel,Q):
    "function"
    import math
    Lij=matrix([[0.0 for row in range(n)]for col in range(Q)])
    DLij=matrix([[0.0 for row in range(n)]for col in range(Q)])
    dLij=matrix([[0.0 for row in range(n)]for col in range(Q)])
    for k in range(Q):#RUNNING THE LOOP FOR FOR CALCULATION OF LAGARNGIAN COEFFICIENTS
        for i in range(n):
           Lij[k,i]=1.0
           DLij[k,i]=1.0
           for j in range(n):
              if i!=j: #CALCULATION OF LAGRANGIAN POLYNOMIAL COEFFICIENETS
                 Lij[k,i]=Lij[k,i]*(z[k]-zcal[j])/(zcal[i]-zcal[j])
              if i!=j:
                 if z[k]-zcal[j]!=0:#DERIVATIVE OF LAGRANGIN COEFFICIENTS i.e BASIS FUNCTIONS
                    DLij[k,i]=DLij[k,i]*(z[k]-zcal[j])/(zcal[i]-zcal[j])
                 else:
                    DLij[k,i]=DLij[k,i]/(zcal[i]-zcal[j])
           indicator=1
           for j in range(n): #CALCULATION OF DERIVATIVE OF LAGRANGIAN POLYNOMIAL VALUE
              if i!=j:
                 if z[k]-zcal[j]==0:
                    dLij[k,i]=DLij[k,i]
                    indicator=0
                    #indicator CHECKS IF INTERPOLATING POINT =DATAPOINT OR NOT
           if indicator!=0:
              for j in range(n):
                 if i!=j:
                    #SUMMATION OF ALL LAGRANGIAN DIFFERENTIAL COEFFICIENTS
                    dLij[k,i]=dLij[k,i]+(DLij[k,i]/(z[k]-zcal[j]))
    #CALCULATION OF INTEGRATION  MASS  AND DIFFERENTIATION
    ## INTILIZATON OF USEFUL VARIABLES
    Int=matrix([[0.0 for row in range(n)]for col in range(n)])
    Int1=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    Dif=matrix([[0.0 for row in range(n)]for col in range(n)])
    Dif1=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    ij=0
    #INTEGRATION OF  TENSOR PRODUCT OF 1D BASIS FUNCTIONS USING GAUSS-NUMERIACL INTEGRATION METHOD
    for i in range(n):
        for j in range(n):
            for k in range(Q):
                Int[i,j]=Int[i,j]+w[k]*Lij[k,i]*Lij[k,j]#LAGRANGIAN POLYNOMIALS
                Int1[ij,0]=Int[i,j]
                Dif[i,j]=Dif[i,j]+w[k]*Lij[k,i]*dLij[k,j]#DIFFERENTIAL OF LAGRANGIAN POLYNOMIAL
                Dif1[ij,0]=Dif[i,j]
            ij=ij+1
    L1=matrix([[0.0 for row in range(n)]for col in range(n)])
    LQ=matrix([[0.0 for row in range(n)]for col in range(n)])
    L11=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    LQQ=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    L1=(Lij[0,:].T*Lij[0,:])# SQUARING ELEMENTS IN FIRST ROW AND SAVING
    LQ=(Lij[Q-1,:].T*Lij[Q-1,:])#SQUARING ELEMENTS IN LAST ROW AND SAVING
    L11=L1.reshape(Qn,1)#"reshape" FUNCTION  CONVERT A MATRIX INTO A ROW MATRIX
    LQQ=LQ.reshape(Qn,1)
    M=numpy.tile(Int,(n,n)) #"tile" FUNCTION CONSTRUCT AN ARRAY BY REPEATING 
    Dx=numpy.tile(Dif,(n,n))#A MATRIX "n" TIME IN BOTH ROW WISE AND COLUMN WISE.
    Dy=numpy.tile(Int,(n,n))#Eg: 1 2
    F1=numpy.tile(Int,(n,n))#    3 4 MATRIX WILL BE CONVERTED INTO
    F2=numpy.tile(LQ,(n,n)) #1 2 1 2
    F3=numpy.tile(Int,(n,n))#3 4 3 4
    F4=numpy.tile(L1,(n,n)) #1 2 1 2
    k1,kk1,ij=0,n,0         #3 4 3 4 IF WE USE "n=2"
    while kk1<=Qn:
        k2,kk2=0,n
        while kk2<=Qn:
            for row in range(k1,kk1):
                for col in range(k2,kk2):
                    M[row,col]=9*Int1[ij,0]*M[row,col]  #2D MASS MATRIX
                    Dy[row,col]=6*Dif1[ij,0]*Dy[row,col]#2D X-DIFFERENTIATION MATRIX
                    Dx[row,col]=6*Int1[ij,0]*Dx[row,col]#2D Y-DIFFERENTIATION MATRIX
                    F1[row,col]=3*L11[ij,0]*F1[row,col] #2D FULX MATRIX OF BOOTOM SIDE
                    F2[row,col]=3*Int1[ij,0]*F2[row,col]#2D FULX MATRIX OF RIGHT SIDE
                    F3[row,col]=3*LQQ[ij,0]*F3[row,col] #2D FULX MATRIX OF TOP SIDE
                    F4[row,col]=3*Int1[ij,0]*F4[row,col]#2D FULX MATRIX OF LEFT SIDE
            k2,kk2,ij=k2+n,kk2+n,ij+1
        k1,kk1=k1+n,kk1+n
    Dx=Dx.T
    Dy=Dy.T
    M,Dx,Dy,F1,F2,F3,F4=(dx*dy/36)*M,(dy/12)*Dx,(dx/12)*Dy,(dx/6)*F1,(dy/6)*F2,(dx/6)*F3,(dy/6)*F4
    I=matrix([[0.0 for row in range(Qn)]for col in range(Qn)])
    for row in range(Qn):
       for col in range(Qn):
          if row==col:
             I[row,col]=1.0
    M_inv=linalg.solve(M, I)# "linalg" GIVES INVERSE IF I S IDENTITY MATRIX
    RDx,RDy,RF1,RF2,RF3,RF4=M_inv*Dx,M_inv*Dy,M_inv*F1,M_inv*F2,M_inv*F3,M_inv*F4
    return M,Dx,Dy,F1,F2,F3,F4,RDx,RDy,RF1,RF2,RF3,RF4,Lij
#################################################################################
#-------------------------------------------------------------------------------#
##THE FOLLOWING FUNCTION CALCULATE GLOBAL COORDINATES ON EACH ELEMENT FOR
## GIVEN LOCAL X AND Y COORDINATES
#-------------------------------------------------------------------------------#
#################################################################################
def global_xy(a_x,a_y,dx,n,Nel,Nopx,Nelxy):
    " a_x and a_y are global coordintaes for the left most and bottom most element"
    " dx= element size, Nel= no.of elemnts in x or y direction"
    " Nelxy=Total no.o elements in the domain"
    " Nopx=no.of interpolted points in x or y direction"
    import math
    x_l=array([[0.0 for col in range(n)]for ele in range(Nel)])#COORDINATES IN X DIRECTION
    y_l=array([[0.0 for col in range(n)]for ele in range(Nel)])#COORDINATES IN Y DIRECTION
    x_d=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    y_d=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    kk,b_x,b_y,Elm=1,0.0,0.0,0
    b_x,b_y=a_x+dx,a_y+dx
    while kk<=Nopx:
       j=0
       for col in range(n):
          x_l[Elm][col]=0.5*(b_x-a_x)*zcal[j]+0.5*(b_x+a_x)
          y_l[Elm][col]=0.5*(b_y-a_y)*zcal[j]+0.5*(b_y+a_y)
          j,kk=j+1,kk+1
       a_x,b_x,a_y,b_y,Elm=b_x,b_x+dx,b_y,b_y+dx,Elm+1
    # FOLLOWING ALGORITHM CALCULTE COORDINTES ON EACH ELEMENT IN BOTH X&Y DIRECTIONS
    for j in range(Nel):
       i,k=j,0
       while i<Nelxy:
          for row in range(n):
             for col in range(n):
                x_d[i][row][col]=x_l[j][col]
                y_d[i][row][col]=y_l[k][row]
          i,k=i+Nel,k+1
    return x_d,y_d,x_l,y_l  
###################################################################
#------------------------------------------------------------------
#The follwing function will calculate Initial and exact solution
#------------------------------------------------------------------
###################################################################
def initial_exact(x_s,y_s,n,Nel,Nop,Nopx,Nelxy):
    "function"
    import math
    xc,yc,sigma,t=-0.0,0.0,1/8.0,1.0
    qi=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    qt=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    u=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    v=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    fu_i=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    fv_i=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    for Elm in range(Nelxy):
       for row in range(n):
          for col in range(n):
             Temp=(x_s[Elm][row][col]-xc)**2+(y_s[Elm][row][col]-yc)**2
             qi[Elm][row][col]=math.exp(-Temp/(2*sigma**2))#Initial Solution
             u[Elm][row][col]=y_s[Elm][row][col]
             fu_i[Elm][row][col]=qi[Elm][row][col]*u[Elm][row][col]
             v[Elm][row][col]=-x_s[Elm][row][col]
             fv_i[Elm][row][col]=qi[Elm][row][col]*v[Elm][row][col]
             xx=x_s[Elm][row][col]-xc*math.cos(t)-yc*math.sin(t)
             yy=y_s[Elm][row][col]+xc*math.sin(t)-yc*math.cos(t)
             qt[Elm][row][col]=math.exp(-(xx**2+yy**2)/(2*sigma**2))#Solution after time "t"
    return u,v,fu_i,fv_i,qi,qt
###################################################################
#-----------------------------------------------------------------#
#The following function will calculate REusanov Fluxes
#-----------------------------------------------------------------#
###################################################################
def rusanov_flux(fu_r,fv_r,q_r,u,v,n,Nel,Nelxy):
    "function"
    import math
    R_Flux=array([[[0.0 for col in range(n)] for row in range(4)] for ele in range(Nelxy)])
    n_1,n_2,n_3,n_4=-1,1,1,-1
    for ele in range(Nelxy):
       for row in range(4):
          for col in range(n):
             if row==0:
                if ele<Nel:
                   Temp=ele+Nel*(Nel-1)
                else:
                   Temp=ele-Nel
                lamb=max(abs(v[ele][0][col]),abs(v[Temp][n-1][col]))
                R_Flux[ele][row][col]=0.5*(n_1*(fv_r[ele][0][col]+fv_r[Temp][n-1][col])-lamb*(q_r[Temp][n-1][col]-q_r[ele][0][col]))
             elif row==1:
                if (ele+1)%Nel==0:
                   Temp=ele-(Nel-1)
                else:
                   Temp=ele+1
                lamb=max(abs(u[ele][col][n-1]),abs(u[Temp][col][0]))
                R_Flux[ele][row][col]=0.5*(n_2*(fu_r[ele][col][n-1]+fu_r[Temp][col][0])-lamb*(q_r[Temp][col][0]-q_r[ele][col][n-1]))
             elif row==2:
                if ele>=Nel*(Nel-1):
                   Temp=ele-Nel*(Nel-1)
                else:
                   Temp=ele+Nel
                lamb=max(abs(v[ele][n-1][col]),abs(v[Temp][0][col]))
                R_Flux[ele][row][col]=0.5*(n_3*(fv_r[ele][n-1][col]+fv_r[Temp][0][col])-lamb*(q_r[Temp][0][col]-q_r[ele][n-1][col]))
             else:
                if ele%Nel==0:
                   Temp=ele+Nel-1
                else:
                   Temp=ele-1
                lamb=max(abs(u[ele][col][0]),abs(u[Temp][col][n-1]))
                R_Flux[ele][row][col]=0.5*(n_4*(fu_r[ele][col][0]+fu_r[Temp][col][n-1])-lamb*(q_r[Temp][col][n-1]-q_r[ele][col][0]))
    return R_Flux
#########################################################################
#-----------------------------------------------------------------------#
########################---RK2 Solver---#################################
#-----------------------------------------------------------------------#
#########################################################################
def RK2_Solver(fu,fv,u,v,R_Flux,n,Nel,Nelxy,Qn,RDx,RDy,RF1,RF2,RF3,RF4,AMR,E_No):
    "function"
    import math
    Flux1=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    Flux2=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    Flux3=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    Flux4=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    fu_n=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    fv_n=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    RHS=matrix([[0.0 for col in range(n)]for row in range(n)])
    Rq=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    for ele in range(Nelxy):
       j=0
       for i in range(n):
          Flux1[i,0]=R_Flux[ele][0][i]
          Flux3[Qn-1-i,0]=R_Flux[ele][2][n-1-i]
          Flux2[n-1+j,0]=R_Flux[ele][1][i]
          Flux4[j,0]=R_Flux[ele][3][i]
          j=j+n
       fu_n,fv_n=matrix(fu[ele]).reshape(Qn,1),matrix(fv[ele]).reshape(Qn,1)
       #print(Flux1,Flux2,Flux3,Flux4)
       RHS=((RDx*fu_n-(RF2*Flux2+RF4*Flux4)+RDy*fv_n-(RF1*Flux1+RF3*Flux3))).reshape(n,n)
       for i in range(n):
          for j in range(n):
             Rq[ele][i][j]=RHS[i,j]
    return Rq
#-----------------------------------------------------------------------#
#The following function will Reshape the given array
#-----------------------------------------------------------------------#
#########################################################################
def Reshape_(q_r,n,Nel,Nelxy,Nopx):
    
    import math
    q_nn=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
    ele,row,col=0,0,0
    for i in range(Nopx):
       for j in range(Nopx):
          q_nn[i][j]=q_r[ele][row][col]
          if (j+1)%n==0:
             col,ele=0,ele+1
          else:
             col=col+1
       if (i+1)%n==0:
          row=0
       else:
          row,ele=row+1,ele-Nel
    return q_nn
#########################################################################
#-----------------------------------------------------------------------#
#The 2D DG solver starts from here
#-----------------------------------------------------------------------#
#########################################################################
print '-------------DG_Gaussian_2D------------ \n'
Nel=int(input('Enter No.of elements required:'))
N=int(input('Enter order of the interpolation required:'))
n=N+1
dx=2.0/Nel
Nopx=Nel*(N+1)
Nopy=Nel*(N+1)
Nop=Nopx*Nopy
Nelxy=Nel**2
dy=dx
Qn=n*n
Qe=Nel*Nel
print'[1]Exact Integration\n'
print'(2]Inexact Integration\n'
Option=int(input('Enter your option fo integration:'))
import time
start_time=time.time()
if Option==1:
    Q=n+1
else:
    Q=n
AMR=0
#CALCULATION OF DATA POINTS USING legendre_gauss_lobatto FUNCTION
[zcal,wcal]=legendre_gauss_lobatto(n);
[z,w]=legendre_gauss_lobatto(Q);
[M,Dx,Dy,F1,F2,F3,F4,RDx,RDy,RF1,RF2,RF3,RF4,Lij]=mass_diff_flux(zcal,wcal,z,w,N,Nel,Q);
# Global coordinates x,y calculation
[x,y,x_l,y_l]= global_xy(-1.0,-1.0,dx,n,Nel,Nopx,Nelxy);
minx=x[0][0][1]-x[0][0][0]
umax=2**0.5
max_dt=minx/(4)
print 'Maximum Time Step:',max_dt
[u,v,fu_i,fv_i,qi,q_t]=initial_exact(x,y,n,Nel,Nop,Nopx,Nelxy);
#[n_x,n_y]=normals(x,y,n,Nel,Nelxy);
q=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn=qi
Rev_T=1#2*math.pi # Time for 1 revolution of gaussian
dt=input('Enter dt:')#dt=0.001 #Time step value
T=int(Rev_T/dt)
E_No=0
#------------------------------------------------------------------------------------------#
q_all=array([[[0.0 for col in range(Nopx)] for row in range(Nopx)] for t_steps in range(T)])
print'Time step used for Rk2 integration:',dt
for Time in range(T):
   R_Flux=rusanov_flux(fu_i,fv_i,qn,u,v,n,Nel,Nelxy);
   #RK2 FIRST STEP
   Rq=RK2_Solver(fu_i,fv_i,u,v,R_Flux,n,Nel,Nelxy,Qn,RDx,RDy,RF1,RF2,RF3,RF4,AMR,E_No);
   q=qn+(0.5*dt*Rq)
   fu_i,fv_i=q*u,q*v# FLUXEX IN BOTH DIRECTIONS
   R_Flux=rusanov_flux(fu_i,fv_i,q,u,v,lambdc_u,lambdc_v,lambdc_V,n,Nel,Nelxy);
   #RK2 SECOND STEP
   Rq=RK2_Solver(fu_i,fv_i,u,v,R_Flux,n,Nel,Nelxy,Qn,RDx,RDy,RF1,RF2,RF3,RF4,AMR,E_No);
   qn=qn+(dt*Rq)
   fu_i,fv_i=qn*u,qn*v# FLUXEX IN BOTH DIRECTIONS
   q_numer=Reshape_(qn,n,Nel,Nelxy,Nopx);
   for row in range(Nopx):
      for col in range(Nopx):
         q_all[Time][row][col]=q_numer[row][col]
   #print(R_Flux)
WCT=time.time()-start_time
print'Wall Clock Time:',WCT
#------------------#-----PLOTING RESULTS-----#------------------#
q_numer=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
q_anlyt=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
#Reshaping the matrices according to the domain indexing for the purpose of drawing plots
q_numer=Reshape_(qn,n,Nel,Nelxy,Nopx);
q_anlyt=Reshape_(q_t,n,Nel,Nelxy,Nopx);
x_l,y_l=x_l.reshape(Nopx,1),y_l.reshape(Nopx,1)
xn,yn=numpy.meshgrid(x_l,y_l)
error=q_numer-q_anlyt
SumE2=numpy.dot(error.reshape(1,Nop),error.reshape(Nop,1))
SumA2=numpy.dot(q_anlyt.reshape(1,Nop),q_anlyt.reshape(Nop,1))
L2=math.log10((SumE2[0][0]/SumA2[0][0])**0.5)
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab as py
'''for i in range(T):
   fig = plt.figure()
   ax = fig.gca(projection='3d')#ax =Axes3D(fig)#ax = fig.add_plot(111, projection='3d')
   surf=ax.plot_surface(xn,yn,q_all[i],rstride=1,cstride=1,cmap=cm.RdBu)
   fileName = "/home/sivakesava/Myfiles/Python/Figures/image-%04d.png" % i
   plt.savefig(fileName, format="png")
   #py.clf()  # Clear the figure for the next loop'''
fig = plt.figure()
ax = fig.gca(projection='3d')#add_subplot(111,projection='3d')#ax =Axes3D(fig)#ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(xn,yn,q_all[T-1],rstride=1, cstride=1,cmap=cm.RdBu)#,linewidth=0)
#rstride=1,cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
         # rstride=6, cstride=6, cmap=cm.jet,cm.RdBu
py.title('q_numerical(t)(x,y)')
py.xlabel('x')
py.ylabel('y')
#cb = fig.colorbar(surf)
plt.show()
##################################################
#---------------------END OF THE CODE---------------------------#
#################################################################
