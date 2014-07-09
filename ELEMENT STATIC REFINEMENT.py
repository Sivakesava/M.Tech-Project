#----------------------------------------------------------------#
#This code Solve Hyperbolic 2D wave PDE using DG methods
#and refines any element on user's choice
#wriiten by Sivakesava Venum for the fulfilment of M.Tech degree
#           Mechanical Engineering Department, IIT BOMBAY
#----------------------------------------------------------------#
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
#######################################################################
#This code computes the Legendre-Gauss-Lobatto points and weights
#which are the roots of the Lobatto Polynomials.
#######################################################################
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
#flux matrices using either exact or iNelxact integration
###################################################################
#-----------------------------------------------------------------#
def mass_diff_flux(zcal,wcal,z,w,N,Nel,Q):
    "zcal array contain Gauss_Lobatto points for give order N, wcal contain their weights"
    "Z array containgauss_lobatto points for integration order Q & w contaion their weights"
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
    #INTEGRATION OF  TENSOR PRODUCT OF 1D BASIS FUNCTIONS USING GAUSS Quadrature INTEGRATION METHOD
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
    return M,Dx,Dy,F1,F2,F3,F4,RDx,RDy,RF1,RF2,RF3,RF4
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
             qt[Elm][row][col]=math.exp(-(xx**2+yy**2)/(2*sigma**2))#Exact Solution
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
def RK2_Solver(fu_i,fv_i,u,v,R_Flux,n,Nel,Nelxy,Qn,RDx,RDy,RF1,RF2,RF3,RF4,AMR,E_No):
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
    fu=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    fv=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    for ele in range(Nelxy):
       j=0
       if AMR==1:
          if ele==E_No:
             ele=ele+1
       for i in range(n):
          Flux1[i,0]=R_Flux[ele][0][i]
          Flux3[Qn-1-i,0]=R_Flux[ele][2][n-1-i]
          Flux2[n-1+j,0]=R_Flux[ele][1][i]
          Flux4[j,0]=R_Flux[ele][3][i]
          j=j+n
       fu_n,fv_n=matrix(fu_i[ele]).reshape(Qn,1),matrix(fv_i[ele]).reshape(Qn,1)
       #print(Flux1,Flux2,Flux3,Flux4)
       RHS=((RDx*fu_n-(RF2*Flux2+RF4*Flux4)+RDy*fv_n-(RF1*Flux1+RF3*Flux3))).reshape(n,n)
       for i in range(n):
          for j in range(n):
             Rq[ele][i][j]=RHS[i,j]
    return Rq
#########################################################################
#########################################################################
#-----------------------------------------------------------------------#
#The following function calculte projection matrices for 1D mortars.
#-----------------------------------------------------------------------#
#########################################################################
def mortar_projection(z,w,zcal,x,y,Nel,n,Q,Lij,):
    "function"
    import math
    Nopx_m=2*n
    Nop_m=Nopx_m**2
    z1=matrix([[0.0 for col in range(1)]for row in range(Q)])
    z2=matrix([[0.0 for col in range(1)]for row in range(Q)])
    z3=matrix([[0.0 for col in range(1)]for row in range(Q)])
    #z4=matrix([[0.0 for row in range(1)]for col in range(Q)])
    for i in range(Q):
       z1[i,0],z2[i,0],z3[i,0]=z[i],(z[i]-1.0)/2,(z[i]+1.0)/2
    Lij1=matrix([[0.0 for col in range(n)]for row in range(Q)])
    Lij2=matrix([[0.0 for col in range(n)]for row in range(Q)])
    Lij3=matrix([[0.0 for col in range(n)]for row in range(Q)])
    #Lij4=matrix([[0.0 for row in range(n)]for col in range(Q)])
    for k in range(Q):#RUNNING THE LOOP FOR FOR CALCULATION OF LAGARNGIAN COEFFICIENTS
       for i in range(n):
          Lij1[k,i],Lij2[k,i],Lij3[k,i]=1.0,1.0,1.0
          for j in range(n):
             if i!=j: #CALCULATION OF LAGRANGIAN POLYNOMIAL VALUE
                Lij1[k,i]=Lij1[k,i]*(z1[k,0]-zcal[j])/(zcal[i]-zcal[j])
                Lij2[k,i]=Lij2[k,i]*(z2[k,0]-zcal[j])/(zcal[i]-zcal[j])
                Lij3[k,i]=Lij3[k,i]*(z3[k,0]-zcal[j])/(zcal[i]-zcal[j])
                #Lij4[k,i]=Lij4[k,i]*(z4[k,0]-zcal[j])/(zcal[i]-zcal[j])
    #CALCULATION OF Intermediate matrices for projecting fluxes from mortars
    S1=matrix([[0.0 for col in range(n)]for row in range(n)])
    S2=matrix([[0.0 for col in range(n)]for row in range(n)])
    Mo=matrix([[0.0 for col in range(n)]for row in range(n)])
    for i in range(n):
       for j in range(n):
          for k in range(Q):
             S1[i,j]=S1[i,j]+w[k]*Lij2[k,j]*Lij1[k,i]
             S2[i,j]=S2[i,j]+w[k]*Lij3[k,j]*Lij1[k,i]
             Mo[i,j]=Mo[i,j]+w[k]*Lij[k,i]*Lij[k,j]
    S1_2,S2_2=0.5*S1.T,0.5*S2.T
    #S=S1+S2
    MS1,MS2,MS1_2,MS2_2=linalg.solve(Mo,S1),linalg.solve(Mo,S2),linalg.solve(Mo,S1_2),linalg.solve(Mo,S2_2)
    return MS1,MS2,MS1_2,MS2_2,Nopx_m,Nop_m,S1,S2,S1_2,S2_2,Mo
###############################################################################
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
###############################################################################
def mortar_sides(E_row,E_col,u,v,n,MS1,MS2):
    "function"
    E_No=(E_row-1)*Nel+E_col-1
    if E_No<Nel:
       BEN=E_No+Nel*(Nel-1)
    else:
       BEN=E_No-Nel
    if (E_No+1)%Nel==0:
       REN=E_No-(Nel-1)
    else:
       REN=E_No+1
    if E_No>=Nel*(Nel-1):
       TEN=E_No-Nel*(Nel-1)
    else:
       TEN=E_No+Nel
    if E_No%Nel==0:
       LEN=E_No+(Nel-1)
    else:
       LEN=E_No-1
    u_l=matrix([[0.0 for col in range(1)]for row in range(n)])
    u_r=matrix([[0.0 for col in range(1)]for row in range(n)])
    v_b=matrix([[0.0 for col in range(1)]for row in range(n)])
    v_t=matrix([[0.0 for col in range(1)]for row in range(n)])
    for i in range(n):
       u_l[i][0]=u[LEN][i][n-1]
       u_r[i][0]=u[REN][i][0]
       v_b[i][0]=v[BEN][n-1][i]
       v_t[i][0]=v[TEN][0][i]
    u_l1,u_l2,u_r1,u_r2=MS1*u_l,MS2*u_l,MS1*u_r,MS2*u_r
    v_b1,v_b2,v_t1,v_t2=MS1*v_b,MS2*v_b,MS1*v_t,MS2*v_t
    return BEN,REN,TEN,LEN,E_No,u_l1,u_l2,u_r1,u_r2,v_b1,v_b2,v_t1,v_t2
################################################################################
#------------------------------------------------------------------------------#
#Flux calculation for non conformal grid
#------------------------------------------------------------------------------#
################################################################################
def mortar_flux(MS1,MS2,MS1_2,MS2_2,qn,u_sub,v_sub,fui_sub,fvi_sub,qn_sub,BEN,REN,TEN,LEN,u_l1,u_l2,u_r1,u_r2,v_b1,v_b2,v_t1,v_t2,R_Flux):
    "function"
    import math
    LEFT=matrix([[0.0 for col in range(1)]for row in range(n)])
    RIGHT=matrix([[0.0 for col in range(1)]for row in range(n)])
    BOTTOM=matrix([[0.0 for col in range(1)]for row in range(n)])
    TOP=matrix([[0.0 for col in range(1)]for row in range(n)])
    fu_l1=matrix([[0.0 for col in range(1)]for row in range(n)])
    fu_l2=matrix([[0.0 for col in range(1)]for row in range(n)])
    fu_r1=matrix([[0.0 for col in range(1)]for row in range(n)])
    fu_r2=matrix([[0.0 for col in range(1)]for row in range(n)])
    fv_b1=matrix([[0.0 for col in range(1)]for row in range(n)])
    fv_b2=matrix([[0.0 for col in range(1)]for row in range(n)])
    fv_t1=matrix([[0.0 for col in range(1)]for row in range(n)])
    fv_t2=matrix([[0.0 for col in range(1)]for row in range(n)])
    for i in range(n):
       LEFT[i][0]=qn[LEN][i][n-1]
       RIGHT[i][0]=qn[REN][i][0]
       BOTTOM[i][0]=qn[BEN][n-1][i]
       TOP[i][0]=qn[TEN][0][i]
    #ax,ay=x[E_col-1][0],y[E_row-1][0]
    #print(BEN,REN,TEN,LEN,BOTTOM,RIGHT,TOP,LEFT)n
    #print ax,ay
    q_b1,q_b2=MS1*BOTTOM,MS2*BOTTOM
    q_r1,q_r2=MS1*RIGHT,MS2*RIGHT
    q_t1,q_t2=MS1*TOP,MS2*TOP
    q_l1,q_l2=MS1*LEFT,MS2*LEFT
    for i in range(n):
       fu_l1[i][0],fu_l2[i][0],fu_r1[i][0],fu_r2[i][0]=u_l1[i][0]*q_l1[i][0],u_l2[i][0]*q_l2[i][0],u_r1[i][0]*q_r1[i][0],u_r2[i][0]*q_r2[i][0]
       fv_b1[i][0],fv_b2[i][0],fv_t1[i][0],fv_t2[i][0]=v_b1[i][0]*q_b1[i][0],v_b2[i][0]*q_b2[i][0],v_t1[i][0]*q_t1[i][0],v_t2[i][0]*q_t2[i][0]
    #Rusanov Flux Calculations
    R_subFlux=array([[[0.0 for col in range(n)] for row in range(4)] for ele in range(4)])
    n_1,n_2,n_3,n_4=-1,1,1,-1
    for ele in range(4):
       for i in range(n):
          if ele==0:
             lamb=max(abs(v_sub[ele][0][i]),abs(v_b1[i][0]))
             R_subFlux[ele][0][i]=0.5*(n_1*(fvi_sub[ele][0][i]+fv_b1[i][0])-lamb*(q_b1[i][0]-qn_sub[ele][0][i]))
             lamb=max(abs(u_sub[ele][i][n-1]),abs(u_sub[1][i][0]))
             R_subFlux[ele][1][i]=0.5*(n_2*(fui_sub[ele][i][n-1]+fui_sub[1][i][0])-lamb*(qn_sub[1][i][0]-qn_sub[ele][i][n-1]))
             lamb=max(abs(v_sub[ele][n-1][i]),abs(v_sub[2][0][i]))
             R_subFlux[ele][2][i]=0.5*(n_3*(fvi_sub[ele][n-1][i]+fvi_sub[2][0][i])-lamb*(qn_sub[2][0][i]-qn_sub[ele][n-1][i]))
             lamb=max(abs(u_sub[ele][i][0]),abs(u_l1[i][0]))
             R_subFlux[ele][3][i]=0.5*(n_4*(fui_sub[ele][i][0]+fu_l1[i][0])-lamb*(q_l1[i][0]-qn_sub[ele][i][0]))
          elif ele==1:
             lamb=max(abs(v_sub[ele][0][i]),abs(v_b2[i][0]))
             R_subFlux[ele][0][i]=0.5*(n_1*(fvi_sub[ele][0][i]+fv_b2[i][0])-lamb*(q_b2[i][0]-qn_sub[ele][0][i]))
             lamb=max(abs(u_sub[ele][i][n-1]),abs(u_r1[i][0]))
             R_subFlux[ele][1][i]=0.5*(n_2*(fui_sub[ele][i][n-1]+fu_r1[i][0])-lamb*(q_r1[i][0]-qn_sub[ele][i][n-1]))
             lamb=max(abs(v_sub[ele][n-1][i]),abs(v_sub[3][0][i]))
             R_subFlux[ele][2][i]=0.5*(n_3*(fvi_sub[ele][n-1][i]+fvi_sub[3][0][i])-lamb*(qn_sub[3][0][i]-qn_sub[ele][n-1][i]))
             lamb=max(abs(u_sub[ele][i][0]),abs(u_sub[0][i][n-1]))
             R_subFlux[ele][3][i]=0.5*(n_4*(fui_sub[ele][i][0]+fui_sub[0][i][n-1])-lamb*(qn_sub[0][i][n-1]-qn_sub[ele][i][0]))
          elif ele==2:
             lamb=max(abs(v_sub[ele][0][i]),abs(v_sub[0][n-1][i]))
             R_subFlux[ele][0][i]=0.5*(n_1*(fvi_sub[ele][0][i]+fvi_sub[0][n-1][i])-lamb*(qn_sub[0][n-1][i]-qn_sub[ele][0][i]))
             lamb=max(abs(u_sub[ele][i][n-1]),abs(u_sub[3][i][0]))
             R_subFlux[ele][1][i]=0.5*(n_2*(fui_sub[ele][i][n-1]+fui_sub[3][i][0])-lamb*(qn_sub[3][i][0]-qn_sub[ele][i][n-1]))
             lamb=max(abs(v_sub[ele][n-1][i]),abs(v_t1[i][0]))
             R_subFlux[ele][2][i]=0.5*(n_3*(fvi_sub[ele][n-1][i]+fv_t1[i][0])-lamb*(q_t1[i][0]-qn_sub[ele][n-1][i]))
             lamb=max(abs(u_sub[ele][i][0]),abs(u_l2[i][0]))
             R_subFlux[ele][3][i]=0.5*(n_4*(fui_sub[ele][i][0]+fu_l2[i][0])-lamb*(q_l2[i][0]-qn_sub[ele][i][0]))
          else: # ele==3
             lamb=max(abs(v_sub[ele][0][i]),abs(v_sub[1][n-1][i]))
             R_subFlux[ele][0][i]=0.5*(n_1*(fvi_sub[ele][0][i]+fvi_sub[1][n-1][i])-lamb*(qn_sub[1][n-1][i]-qn_sub[ele][0][i]))
             lamb=max(abs(u_sub[ele][i][n-1]),abs(u_r2[i][0]))
             R_subFlux[ele][1][i]=0.5*(n_2*(fui_sub[ele][i][n-1]+fu_r2[i][0])-lamb*(q_r2[i][0]-qn_sub[ele][i][n-1]))
             lamb=max(abs(v_sub[ele][n-1][i]),abs(v_t2[i][0]))
             R_subFlux[ele][2][i]=0.5*(n_3*(fvi_sub[ele][n-1][i]+fv_t2[i][0])-lamb*(q_t2[i][0]-qn_sub[ele][n-1][i]))
             lamb=max(abs(u_sub[ele][i][0]),abs(u_sub[2][i][n-1]))
             R_subFlux[ele][3][i]=0.5*(n_4*(fui_sub[ele][i][0]+fui_sub[2][i][n-1])-lamb*(qn_sub[2][i][n-1]-qn_sub[ele][i][0]))
             
    R_Flux[BEN][2]=array((MS1_2*(matrix(-R_subFlux[0][0]).T)+MS2_2*(matrix(-R_subFlux[1][0]).T)).T)
    R_Flux[REN][3]=array((MS1_2*(matrix(-R_subFlux[1][1]).T)+MS2_2*(matrix(-R_subFlux[3][1]).T)).T)
    R_Flux[TEN][0]=array((MS1_2*(matrix(-R_subFlux[2][2]).T)+MS2_2*(matrix(-R_subFlux[3][2]).T)).T)
    R_Flux[LEN][1]=array((MS1_2*(matrix(-R_subFlux[0][3]).T)+MS2_2*(matrix(-R_subFlux[2][3]).T)).T)
    return R_subFlux,R_Flux
#-----------------------------------------------------------------------#
#The following function will Reshape the given array
#-----------------------------------------------------------------------#
#########################################################################
def Reshape_(qn,n,Nel,Nelxy,Nopx):
    "This function reshapes the 3D solution array into 2D array for plotting convenience"
    import math
    q_nn=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
    ele,row,col=0,0,0
    for i in range(Nopx):
       for j in range(Nopx):
          q_nn[i][j]=qn[ele][row][col]
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
print '-------------ELEMENT STATIC REFINEMENT---------------- \n'
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
AMR=int(input('Do you want REFINEMENT (If "Yes" Enter 1, If "No", Enter other than 1):'))
if AMR==1:
   E_row,E_col=int(input('Enter row and column numbers of the element to refine:')),int(input())
import time
start_time=time.time()
if Option==1:
    Q=(n+1)
else:
    Q=n
#CALCULATION OF DATA POINTS USING legendre_gauss_lobatto FUNCTION
[zcal,wcal]=legendre_gauss_lobatto(n);
[z,w]=legendre_gauss_lobatto(Q);
[M,Dx,Dy,F1,F2,F3,F4,RDx,RDy,RF1,RF2,RF3,RF4,Lij]=mass_diff_flux(zcal,wcal,z,w,N,Nel,Q);
# Global coordinates x,y calculation
[x,y,x_l,y_l]= global_xy(-1.0,-1.0,dx,n,Nel,Nopx,Nelxy);
[u,v,fu_i,fv_i,qi,q_t]=initial_exact(x,y,n,Nel,Nop,Nopx,Nelxy);
q=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn=qi
Rev_T=1#0.25*2*math.pi # Time for 1 revolution of gaussian
dt=1e-3
T=int(Rev_T/dt)
#------------------------------------------------------------------------------------------#
if AMR==1: #Refinement is asked for a particular element
   a_x,a_y=x_l[E_col-1][0],y_l[E_row-1][0]
   [MS1,MS2,MS1_2,MS2_2,Nopx_m,Nop_m,S1,S2,S1_20,S2_2,Mo]=mortar_projection(z,w,zcal,x,y,Nel,n,Q,Lij,);
   [BEN,REN,TEN,LEN,E_No,u_l1,u_l2,u_r1,u_r2,v_b1,v_b2,v_t1,v_t2]=mortar_sides(E_row,E_col,u,v,n,MS1,MS2);
   [x_sub,y_sub,x_lsub,y_lsub]= global_xy(a_x,a_y,dx/2,n,2,Nopx_m,4);
   minx=abs(x_lsub[0][1]-x_lsub[0][0])
   umax=2**0.5
   max_dt=minx/(4*umax)
   [u_sub,v_sub,fui_sub,fvi_sub,qi_sub,qt_sub]=initial_exact(x_sub,y_sub,n,2,Nop_m,Nopx_m,4);
   q_sub=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(4)])
   qn_sub=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(4)])
   qn_sub=qi_sub
else:
   E_No=0
   minx=abs(x_l[0][1]-x_l[0][0])
   umax=2**0.5
   max_dt=minx/(4*umax)
print 'Maximum Time Step:',max_dt
#------------------------------------------------------------------------------------------#
print'Time step used for Rk2 integration:',dt
for Time in range(T):
   R_Flux=rusanov_flux(fu_i,fv_i,qn,u,v,n,Nel,Nelxy);#flux calculation
   if AMR==1:
      [R_subFlux,R_Flux]=mortar_flux(MS1,MS2,MS1_2,MS2_2,qn,u_sub,v_sub,fui_sub,fvi_sub,qn_sub,BEN,REN,TEN,LEN,u_l1,u_l2,u_r1,u_r2,v_b1,v_b2,v_t1,v_t2,R_Flux);
      Rq_sub=RK2_Solver(fui_sub,fvi_sub,u_sub,v_sub,R_subFlux,n,2,4,Qn,RDx,RDy,RF1,RF2,RF3,RF4,0,0);
      q_sub=qn_sub+(0.5*dt*2*Rq_sub)#RK2 step1
      fui_sub,fvi_sub=q_sub*u_sub,q_sub*v_sub
   Rq=RK2_Solver(fu_i,fv_i,u,v,R_Flux,n,Nel,Nelxy,Qn,RDx,RDy,RF1,RF2,RF3,RF4,AMR,E_No);
   q=qn+(0.5*dt*Rq)
   fu_i,fv_i=q*u,q*v
   #print(R_Flux)
   R_Flux=rusanov_flux(fu_i,fv_i,q,u,v,n,Nel,Nelxy);
   if AMR==1:
      [R_subFlux,R_Flux]=mortar_flux(MS1,MS2,MS1_2,MS2_2,q,u_sub,v_sub,fui_sub,fvi_sub,q_sub,BEN,REN,TEN,LEN,u_l1,u_l2,u_r1,u_r2,v_b1,v_b2,v_t1,v_t2,R_Flux);
      Rq_sub=RK2_Solver(fui_sub,fvi_sub,u_sub,v_sub,R_subFlux,n,2,4,Qn,RDx,RDy,RF1,RF2,RF3,RF4,0,0);
      qn_sub=qn_sub+(dt*2*Rq_sub)
      fui_sub,fvi_sub=qn_sub*u_sub,qn_sub*v_sub
   Rq=RK2_Solver(fu_i,fv_i,u,v,R_Flux,n,Nel,Nelxy,Qn,RDx,RDy,RF1,RF2,RF3,RF4,AMR,E_No);
   qn=qn+(dt*Rq)#RK2 step2
   fu_i,fv_i=qn*u,qn*v
   #print(R_Flux)
WCT=time.time()-start_time
print'Wall Clock Time:',WCT
#------------------#-----PLOTING RESULTS-----#------------------#
if AMR==1:
   qn[E_No]=None
   q_t[E_No]=None
q_numer=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
q_anlyt=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
#Reshaping the matrices according to the domain indexing for the purpose of drawing plots
q_numer=Reshape_(qn,n,Nel,Nelxy,Nopx);
q_anlyt=Reshape_(q_t,n,Nel,Nelxy,Nopx);
if AMR==1:
   q_numsub=Reshape_(qn_sub,n,2,4,Nopx_m);
   q_anlsub=Reshape_(qt_sub,n,2,4,Nopx_m);
   x_lsub,y_lsub=x_lsub.reshape(Nopx_m,1),y_lsub.reshape(Nopx_m,1)
   xn_sub,yn_sub=numpy.meshgrid(x_lsub,y_lsub)
x_l,y_l=x_l.reshape(Nopx,1),y_l.reshape(Nopx,1)
xn,yn=numpy.meshgrid(x_l,y_l)
'''if AMR==1:
   qn[E_No]=10e5
   q_t[E_No]=10e5
   q_num=Reshape_(qn,n,Nel,Nelxy,Nopx);
   q_any=Reshape_(q_t,n,Nel,Nelxy,Nopx);
   x_all=numpy.append(xn.reshape(Nop,1),xn_sub.reshape(Nop_m,1),axis=0)
   y_all=numpy.append(yn.reshape(Nop,1),yn_sub.reshape(Nop_m,1),axis=0)
   q_all=numpy.append(q_num.reshape(Nop,1),q_numsub.reshape(Nop_m,1),axis=0)
   qt_all=numpy.append(q_any.reshape(Nop,1),q_anlsub.reshape(Nop_m,1),axis=0)
   qn_f=array([0.0 for i in range(Nop+Nop_m-Qn)])
   qt_f=array([0.0 for i in range(Nop+Nop_m-Qn)])
   xn_f=array([0.0 for i in range(Nop+Nop_m-Qn)])
   yn_f=array([0.0 for i in range(Nop+Nop_m-Qn)])
   j=0
   for i in range(Nop+Nop_m-Qn):
       if(q_all[j][0]==10e5):
           j=j+1
       else:
           qn_f[i]=q_all[j][0]
           qt_f[i]=qt_all[j][0]
           xn_f[i]=x_all[j][0]
           yn_f[i]=y_all[j][0]
       j=j+1
   error=qn_f-qt_f
   SumE2=numpy.dot(numpy.transpose(error),error)
   SumA2=numpy.dot(numpy.transpose(qt_f),qt_f)
   L2=math.log10((SumE2/SumA2)**0.5)
   if Option==1:
      fo = open("L2NormsAMRExact.txt", "a+") #Saving error norms in afile
   else:
      fo = open("L2NormsAMRInexact.txt", "a+")
   fo.write(str(Nel));
   fo.write("\t\t");
   fo.write(str(N));
   fo.write("\t\t")
   fo.write(str(Nop+Nop_m-Qn));
   fo.write("\t");
   fo.write(str(L2));
   fo.write("\t\t");
   fo.write(str(WCT));
   fo.write("\n");
   fo.close()
   fo = open("data.txt", "w")
   for i in range(Nop+Nop_m):
      if(q_all[i][0]==10e5):
         i=i+1
      else:
         fo.write(str(x_all[i][0])); #Saving solution in a file
         fo.write("\t\t");
         fo.write(str(y_all[i][0]));
         fo.write("\t\t")
         fo.write(str(q_all[i][0]));
         fo.write("\n");
   # Close opend file
   fo.close()
else:
   error=q_numer-q_anlyt
   SumE2=numpy.dot(error.reshape(1,Nop),error.reshape(Nop,1))
   SumA2=numpy.dot(q_anlyt.reshape(1,Nop),q_anlyt.reshape(Nop,1))
   L2=math.log10((SumE2[0][0]/SumA2[0][0])**0.5)
   if Option==1:
      fo = open("L2NormsExact.txt", "a+") #Saving error norms in afile
   else:
      fo = open("L2NormsInexact.txt", "a+")
   fo.write(str(Nel));
   fo.write("\t\t");
   fo.write(str(N));
   fo.write("\t\t")
   fo.write(str(Nop));
   fo.write("\t");
   fo.write(str(L2));
   fo.write("\t\t");
   fo.write(str(WCT));
   fo.write("\n");
   fo.close()'''
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab as py
fig = plt.figure()
ax = fig.gca(projection='3d')#ax =Axes3D(fig)#ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(xn,yn,q_numer,rstride=1,cstride=1,cmap=cm.RdBu)
if AMR==1:
   surf=ax.plot_surface(xn_sub,yn_sub,q_numsub,rstride=1,cstride=1,cmap=cm.RdBu)
py.title('q_numerical(t)(x,y)')
py.xlabel('x')
py.ylabel('y')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')#ax =Axes3D(fig)
qtplot=ax.plot_surface(xn,yn,q_anlyt,rstride=1,cstride=1,cmap=cm.RdBu)
if AMR==1:
   qtplot=ax.plot_surface(xn_sub,yn_sub,q_anlsub,rstride=1,cstride=1,cmap=cm.RdBu)
py.title('q_analytical(t)(x,y)')
py.xlabel('x')
py.ylabel('y')
plt.show()
'''ax.set_zlim3d(0, q_max)                    # viewrange for z-axis should be [-4,4] 
      ax.set_ylim3d(-1, 1)                    # viewrange for y-axis should be [-2,2] 
      ax.set_xlim3d(-1, 1)                    # viewrange for x-axis should be [-2,2] '''
#################################################################
#---------------------END OF THE CODE---------------------------#
#################################################################
