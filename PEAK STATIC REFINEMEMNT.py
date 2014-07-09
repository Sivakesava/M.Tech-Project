#-----------------------------------------------------------------------------------------------------------#
#THIS CODE STATTICALLY REFINES THE ALL ELEMENTS CONTAING GAUSSIAN CENTRE AT ANY TIME
#FIRST CALCULTES THE ELEMNTS THROUGH WHICH CENTRE PASSES AND THEN REFINEMENT STARTS FROM THE 1ST TIME STEP.
#wriiten by Sivakesava Venum for the fulfilment of M.Tech degree
#           Mechanical Engineering Department, IIT BOMBAY
#------------------------------------------------------------------------------------------------------------#
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
################################################################
#This code computes the Legendre-Gauss-Lobatto points and weights
#which are the roots of the Lobatto Polynomials.
#################################################################
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
def exact(x_s,y_s,n,Nel,Nop,Nopx,Nelxy,t):
    "function"
    import math
    xc,yc,sigma=-0.5,0.0,1/8.0
    q_e=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    u=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    v=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
    for Elm in range(Nelxy):
       for row in range(n):
          for col in range(n):
             u[Elm][row][col]=y_s[Elm][row][col]
             v[Elm][row][col]=-x_s[Elm][row][col]
             xx=x_s[Elm][row][col]-xc*math.cos(t)-yc*math.sin(t)
             yy=y_s[Elm][row][col]+xc*math.sin(t)-yc*math.cos(t)
             q_e[Elm][row][col]=math.exp(-(xx**2+yy**2)/(2*sigma**2))#Exact Solution
    if t==0:
        return q_e,u,v
    else:
        return q_e
###################################################################
#-----------------------------------------------------------------#
#The following function will calculate REusanov Fluxes
#-----------------------------------------------------------------#
###################################################################
def rusanov_flux(fu_r,fv_r,q_r,u,v,n,Nel,Nelxy,boundary_e):
    "function"
    import math
    R_Flux=array([[[0.0 for col in range(n)] for row in range(4)] for ele in range(Nelxy)])
    n_1,n_2,n_3,n_4=-1,1,1,-1
    boundary_e=boundary_e_ele_num(Nel,Nelxy);
    for ele in range(Nelxy):
       for row in range(4):
          for col in range(n):
             if row==0:#bottom flux
                lamb=max(abs(v[ele][0][col]),abs(v[boundary_e[ele][0]][n-1][col]))
                R_Flux[ele][row][col]=0.5*(n_1*(fv_r[ele][0][col]+fv_r[boundary_e[ele][0]][n-1][col])-lamb*(q_r[boundary_e[ele][0]][n-1][col]-q_r[ele][0][col]))
             elif row==1:#left side
                lamb=max(abs(u[ele][col][n-1]),abs(u[boundary_e[ele][1]][col][0]))
                R_Flux[ele][row][col]=0.5*(n_2*(fu_r[ele][col][n-1]+fu_r[boundary_e[ele][1]][col][0])-lamb*(q_r[boundary_e[ele][1]][col][0]-q_r[ele][col][n-1]))
             elif row==2:#top side
                lamb=max(abs(v[ele][n-1][col]),abs(v[boundary_e[ele][2]][0][col]))
                R_Flux[ele][row][col]=0.5*(n_3*(fv_r[ele][n-1][col]+fv_r[boundary_e[ele][2]][0][col])-lamb*(q_r[boundary_e[ele][2]][0][col]-q_r[ele][n-1][col]))
             else:#row=3 #left side
                lamb=max(abs(u[ele][col][0]),abs(u[boundary_e[ele][3]][col][n-1]))
                R_Flux[ele][row][col]=0.5*(n_4*(fu_r[ele][col][0]+fu_r[boundary_e[ele][3]][col][n-1])-lamb*(q_r[boundary_e[ele][3]][col][n-1]-q_r[ele][col][0]))
    return R_Flux
#########################################################################
#--------------boundary_e elements for each element-----------------------#
def boundary_e_ele_num(Nel,Nelxy):
    "function saves the index of elemens surrounding each element"
    boundary_e=array([[0 for side in range(4)]for ele in range(Nelxy)])
    for ele in range(Nelxy):
       if ele<Nel: #Bottom
          boundary_e[ele][0]=ele+Nel*(Nel-1)
       else:
          boundary_e[ele][0]=ele-Nel
       if (ele+1)%Nel==0:#right
          boundary_e[ele][1]=ele-(Nel-1)
       else:
          boundary_e[ele][1]=ele+1
       if ele>=Nel*(Nel-1):#Top
          boundary_e[ele][2]=ele-Nel*(Nel-1)
       else:
          boundary_e[ele][2]=ele+Nel
       if ele%Nel==0:# left
          boundary_e[ele][3]=ele+Nel-1
       else:
          boundary_e[ele][3]=ele-1
    return boundary_e
#########################################################################
#########################################################################
#-----------------------------------------------------------------------#
#The below function calcultes 1D and 2D projection matriecs for Non conformal Grid
#-----------------------------------------------------------------------#
#########################################################################
def projection_matrices(z,w,zcal,x,y,Nel,n,Q):
    "function"
    import math
    #Nopx_m, Nop_m are no. of points in x or y direction after dividing
    #parent element into 4 child elements
    Nopx_m=2*n
    Nop_m=Nopx_m**2
    z1=matrix([[0.0 for col in range(1)]for row in range(Q)])
    z2=matrix([[0.0 for col in range(1)]for row in range(Q)])
    z3=matrix([[0.0 for col in range(1)]for row in range(Q)])
    for i in range(Q):
       #New relative coordinates to evaluate integration using Gauss QUADRATURE
       #This is because in Gauss Quadrature method the limits always should be[-1,+1]
       z1[i,0],z2[i,0],z3[i,0]=z[i],(z[i]-1.0)/2,(z[i]+1.0)/2
    Lij1=matrix([[0.0 for col in range(n)]for row in range(Q)])
    Lij2=matrix([[0.0 for col in range(n)]for row in range(Q)])
    Lij3=matrix([[0.0 for col in range(n)]for row in range(Q)])
    #Lij4=matrix([[0.0 for row in range(n)]for col in range(Q)])
    for k in range(Q):#RUNNING THE LOOP FOR FOR CALCULATION OF LAGARNGIAN COEFFICIENTS
       for i in range(n):
          Lij1[k,i],Lij2[k,i],Lij3[k,i]=1.0,1.0,1.0
          for j in range(n):
             if i!=j: #CALCULATION OF LAGRANGIAN COEFFICIENTS FOR EACH COORDINATE
                Lij1[k,i]=Lij1[k,i]*(z1[k,0]-zcal[j])/(zcal[i]-zcal[j])
                Lij2[k,i]=Lij2[k,i]*(z2[k,0]-zcal[j])/(zcal[i]-zcal[j])
                Lij3[k,i]=Lij3[k,i]*(z3[k,0]-zcal[j])/(zcal[i]-zcal[j])
    #CALCULATION OF Intermediate matrices for projecting fluxes from mortars
    S1=matrix([[0.0 for col in range(n)]for row in range(n)])
    S2=matrix([[0.0 for col in range(n)]for row in range(n)])
    Mo=matrix([[0.0 for col in range(n)]for row in range(n)])
    I1=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    I2=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    I0=matrix([[0.0 for row in range(1)]for col in range(Qn)])
    ij=0
    for i in range(n):      #1D INTEGRATION USING GAUSS Quadrature METHOD
       for j in range(n):   # 2 PROJECTION MATRICES WILL BE THERE IN 1D PROJECTION
          for k in range(Q):
             S1[i,j]=S1[i,j]+w[k]*Lij2[k,j]*Lij1[k,i]#1D BOTTOM projection Matrix
             I1[ij,0]=S1[i,j]
             S2[i,j]=S2[i,j]+w[k]*Lij3[k,j]*Lij1[k,i]#1D  TOP projection Matrix
             I2[ij,0]=S2[i,j]
             Mo[i,j]=Mo[i,j]+w[k]*Lij1[k,i]*Lij1[k,j]#1D projection Mass Matrix
             I0[ij,0]=Mo[i,j]
          ij=ij+1
    S1_2,S2_2=0.5*S1.T,0.5*S2.T
    #2d projection
    MM=numpy.tile(Mo,(n,n)) #"tile" FUNCTION CONSTRUCT AN ARRAY BY REPEATING 
    SS1=numpy.tile(S1,(n,n))#A MATRIX "n" TIME IN BOTH ROW WISE AND COLUMN WISE.
    SS2=numpy.tile(S2,(n,n))#Eg: 1 2
    SS3=numpy.tile(S1,(n,n))#    3 4 MATRIX WILL BE CONVERTED INTO
    SS4=numpy.tile(S2,(n,n))#1 2 1 2
    k1,kk1,ij=0,n,0         #3 4 3 4
    while kk1<=Qn:          #1 2 1 2
        k2,kk2=0,n          #3 4 3 4 IF WE USE "n=2"
        while kk2<=Qn:
            for row in range(k1,kk1):
                for col in range(k2,kk2):
                    MM[row,col]=I0[ij,0]*MM[row,col]  #2D PROJECTION MASS MATRIX
                    SS1[row,col]=I1[ij,0]*SS1[row,col]#2D PROJECTION MATRIX FOR LEFT-BOTTOM CHILD ELEMENT
                    SS2[row,col]=I1[ij,0]*SS2[row,col]#2D PROJECTION MATRIX FOR LEFT-TOP CHILD ELEMENT
                    SS3[row,col]=I2[ij,0]*SS3[row,col]#2D PROJECTION MATRIX FOR RIGHT-BOTTOM CHILD ELEMENT
                    SS4[row,col]=I2[ij,0]*SS4[row,col]#2D PROJECTION MATRIX FOR RIGHT-TOP CHILD ELEMENT
            k2,kk2,ij=k2+n,kk2+n,ij+1
        k1,kk1=k1+n,kk1+n
    MS1,MS2,MS1_2,MS2_2=linalg.solve(Mo,S1),linalg.solve(Mo,S2),linalg.solve(Mo,S1_2),linalg.solve(Mo,S2_2)
    MMS1,MMS2,MMS3,MMS4=linalg.solve(MM,SS1),linalg.solve(MM,SS2),linalg.solve(MM,SS3),linalg.solve(MM,SS4)
    return MS1,MS2,MS1_2,MS2_2,Nopx_m,Nop_m,S1,S2,S1_2,S2_2,Mo,MM,SS1,SS2,SS3,SS4
########################################################################
########################################################################
def normal_flux(q_s,fu_sub,fv_sub,u_sub,v_sub,q_,u,v,fu,fv,Nelxy,n,status,boundary,sub_n,n_sub,MS1,MS2,MS1_2,MS2_2):
    " This unction claculate normal flux on eacg edge"
    r_flux=array([[[0.0 for value in range(n)]for side in range(4)]for ele in range(Nelxy)])
    r_flux_sub=array([[[[0.0 for value in range(n)]for side in range(4)]for ele in range(4)]for sub in range(n_sub)])
    n1,n2,n3,n4=-1,1,1,-1
    for E_no in range(Nelxy):
       if status[E_no]==1:#cheking whether element is marked for refinemen or not
          index_=numpy.where(sub_n==E_no)[0][0]
          if status[boundary[E_no][0]]==1:#cheking whether bottom element is marked for refinemen or not
             index_0=numpy.where(sub_n==boundary[E_no][0])[0][0]
             for i in range(n):
                lmbd=max(abs(v_sub[index_][0][0][i]),abs(v_sub[index_0][2][n-1][i]))#flux calculations
                r_flux_sub[index_][0][0][i]=0.5*(n1*(fv_sub[index_][0][0][i]+fv_sub[index_0][2][n-1][i])-lmbd*(q_s[index_0][2][n-1][i]-q_s[index_][0][0][i]))
                lmbd=max(abs(v_sub[index_][1][0][i]),abs(v_sub[index_0][3][n-1][i]))
                r_flux_sub[index_][1][0][i]=0.5*(n1*(fv_sub[index_][1][0][i]+fv_sub[index_0][3][n-1][i])-lmbd*(q_s[index_0][3][n-1][i]-q_s[index_][1][0][i]))
          else: #bottom element is not marked for refinement
             v_1,v_2=array(MS1*(numpy.matrix(v[boundary[E_no][0],n-1,:]).T)),array(MS2*(numpy.matrix(v[boundary[E_no][0],n-1,:]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[boundary[E_no][0],n-1,:]).T)),array(MS2*(numpy.matrix(q_[boundary[E_no][0],n-1,:]).T))
             for i in range(n):
                lmbd=max(abs(v_sub[index_][0][0][i]),abs(v_1[i][0]))
                r_flux_sub[index_][0][0][i]=0.5*(n1*(fv_sub[index_][0][0][i]+q_1[i][0]*v_1[i][0])-lmbd*(q_1[i][0]-q_s[index_][0][0][i]))
                lmbd=max(abs(v_sub[index_][1][0][i]),abs(v_2[i][0]))
                r_flux_sub[index_][1][0][i]=0.5*(n1*(fv_sub[index_][1][0][i]+q_2[i][0]*v_2[i][0])-lmbd*(q_2[i][0]-q_s[index_][1][0][i]))
          if status[boundary[E_no][1]]==1:#cheking whether right element is marked for refinemen or not
             index_0=numpy.where(sub_n==boundary[E_no][1])[0][0]
             for i in range(n):
                lmbd=max(abs(u_sub[index_][1][i][n-1]),abs(u_sub[index_0][0][i][0]))#flux calculations
                r_flux_sub[index_][1][1][i]=0.5*(n2*(fu_sub[index_][1][i][n-1]+fu_sub[index_0][0][i][0])-lmbd*(q_s[index_0][0][i][0]-q_s[index_][1][i][n-1]))
                lmbd=max(abs(u_sub[index_][3][i][n-1]),abs(u_sub[index_0][2][i][0]))
                r_flux_sub[index_][3][1][i]=0.5*(n2*(fu_sub[index_][3][i][n-1]+fu_sub[index_0][2][i][0])-lmbd*(q_s[index_0][2][i][0]-q_s[index_][3][i][n-1]))
          else:
             u_1,u_2=array(MS1*(numpy.matrix(u[boundary[E_no][1],:,0]).T)),array(MS2*(numpy.matrix(u[boundary[E_no][1],:,0]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[boundary[E_no][1],:,0]).T)),array(MS2*(numpy.matrix(q_[boundary[E_no][1],:,0]).T))
             for i in range(n):
                lmbd=max(abs(u_sub[index_][1][i][n-1]),abs(u_1[i][0]))
                r_flux_sub[index_][1][1][i]=0.5*(n2*(fu_sub[index_][1][i][n-1]+q_1[i][0]*u_1[i][0])-lmbd*(q_1[i][0]-q_s[index_][1][i][n-1]))
                lmbd=max(abs(u_sub[index_][3][i][n-1]),abs(u_2[i][0]))
                r_flux_sub[index_][3][1][i]=0.5*(n2*(fu_sub[index_][3][i][n-1]+q_2[i][0]*u_2[i][0])-lmbd*(q_2[i][0]-q_s[index_][3][i][n-1]))
          if (status[boundary[E_no][2]]==1):#cheking whether top element is marked for refinemen or not
             index_0=numpy.where(sub_n==boundary[E_no][2])[0][0]
             for i in range(n):
                lmbd=max(abs(v_sub[index_][2][n-1][i]),abs(v_sub[index_0][0][0][i]))#flux calculations
                r_flux_sub[index_][2][2][i]=0.5*(n3*(fv_sub[index_][2][n-1][i]+fv_sub[index_0][0][0][i])-lmbd*(q_s[index_0][0][0][i]-q_s[index_][2][n-1][i]))
                lmbd=max(abs(v_sub[index_][3][n-1][i]),abs(v_sub[index_0][1][0][i]))
                r_flux_sub[index_][3][2][i]=0.5*(n3*(fv_sub[index_][3][n-1][i]+fv_sub[index_0][1][0][i])-lmbd*(q_s[index_0][1][0][i]-q_s[index_][3][n-1][i]))
          else:
             v_1,v_2=array(MS1*(numpy.matrix(v[boundary[E_no][2],0,:]).T)),array(MS2*(numpy.matrix(v[boundary[E_no][2],0,:]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[boundary[E_no][2],0,:]).T)),array(MS2*(numpy.matrix(q_[boundary[E_no][2],0,:]).T))
             for i in range(n):
                lmbd=max(abs(v_sub[index_][2][n-1][i]),abs(v_1[i][0]))
                r_flux_sub[index_][2][2][i]=0.5*(n3*(fv_sub[index_][2][n-1][i]+q_1[i][0]*v_1[i][0])-lmbd*(q_1[i][0]-q_s[index_][2][n-1][i]))
                lmbd=max(abs(v_sub[index_][3][n-1][i]),abs(v_2[i][0]))
                r_flux_sub[index_][3][2][i]=0.5*(n3*(fv_sub[index_][3][n-1][i]+q_2[i][0]*v_2[i][0])-lmbd*(q_2[i][0]-q_s[index_][3][n-1][i]))
          if status[boundary[E_no][3]]==1:#cheking whether left element is marked for refinemen or not
             index_0=numpy.where(sub_n==boundary[E_no][3])[0][0]
             for i in range(n):
                lmbd=max(abs(u_sub[index_][0][i][0]),abs(u_sub[index_0][1][i][n-1]))#flux calculations
                r_flux_sub[index_][0][3][i]=0.5*(n4*(fu_sub[index_][0][i][0]+fu_sub[index_0][1][i][n-1])-lmbd*(q_s[index_0][1][i][n-1]-q_s[index_][0][i][0]))
                lmbd=max(abs(u_sub[index_][2][i][0]),abs(u_sub[index_0][3][i][n-1]))
                r_flux_sub[index_][2][3][i]=0.5*(n4*(fu_sub[index_][2][i][0]+fu_sub[index_0][3][i][n-1])-lmbd*(q_s[index_0][3][i][n-1]-q_s[index_][2][i][0]))
          else:
             u_1,u_2=array(MS1*(numpy.matrix(u[boundary[E_no][3],:,n-1]).T)),array(MS2*(numpy.matrix(u[boundary[E_no][3],:,n-1]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[boundary[E_no][3],:,n-1]).T)),array(MS2*(numpy.matrix(q_[boundary[E_no][3],:,n-1]).T))
             for i in range(n):
                lmbd=max(abs(u_sub[index_][0][i][0]),abs(u_1[i][0]))
                r_flux_sub[index_][0][3][i]=0.5*(n4*(fu_sub[index_][0][i][0]+q_1[i][0]*u_1[i][0])-lmbd*(q_1[i][0]-q_s[index_][0][i][0]))
                lmbd=max(abs(u_sub[index_][2][i][0]),abs(u_2[i][0]))
                r_flux_sub[index_][2][3][i]=0.5*(n4*(fu_sub[index_][2][i][0]+q_2[i][0]*u_2[i][0])-lmbd*(q_2[i][0]-q_s[index_][2][i][0]))
          for i in range(4):
             for k in range(n):
                if (i==2 or i==3):
                    lamb=max(abs(v_sub[index_][i][0][k]),abs(v_sub[index_][i-2][n-1][k]))
                    r_flux_sub[index_][i][0][k]=0.5*(n1*(fv_sub[index_][i][0][k]+fv_sub[index_][i-2][n-1][k])-lamb*(q_sub[index_][i-2][n-1][k]-q_sub[index_][i][0][k]))
                if (i==0 or i==2):
                    lamb=max(abs(u_sub[index_][i][k][n-1]),abs(u_sub[index_][i+1][k][0]))
                    r_flux_sub[index_][i][1][k]=0.5*(n2*(fu_sub[index_][i][k][n-1]+fu_sub[index_][i+1][k][0])-lamb*(q_sub[index_][i+1][k][0]-q_sub[index_][i][k][n-1]))
                if (i==0 or i==1):
                    lamb=max(abs(v_sub[index_][i][n-1][k]),abs(v_sub[index_][i+2][0][k]))
                    r_flux_sub[index_][i][2][k]=0.5*(n3*(fv_sub[index_][i][n-1][k]+fv_sub[index_][i+2][0][k])-lamb*(q_sub[index_][i+2][0][k]-q_sub[index_][i][n-1][k]))
                if (i==1 or i==3):
                    lamb=max(abs(u_sub[index_][i][k][0]),abs(u_sub[index_][i-1][k][n-1]))
                    r_flux_sub[index_][i][3][k]=0.5*(n4*(fu_sub[index_][i][k][0]+fu_sub[index_][i-1][k][n-1])-lamb*(q_sub[index_][i-1][k][n-1]-q_sub[index_][i][k][0]))

       else:#Status=0 i.e, element is not marked for refinement
          r_f_1=array([0.0 for i in range(n)])
          r_f_2=array([0.0 for i in range(n)])
          if status[boundary[E_no][0]]==1:#cheking whether bottom element is marked for refinemen or not
             index_=numpy.where(sub_n==boundary[E_no][0])[0][0]
             v_1,v_2=array(MS1*(numpy.matrix(v[E_no,0,:]).T)),array(MS2*(numpy.matrix(v[E_no,0,:]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[E_no,0,:]).T)),array(MS2*(numpy.matrix(q_[E_no,0,:]).T))
             for i in range(n):
                lmbd=max(abs(v_sub[index_][2][n-1][i]),abs(v_1[i][0]))
                r_f_1[i]=0.5*(n1*(fv_sub[index_][2][n-1][i]+q_1[i][0]*v_1[i][0])-lmbd*(q_s[index_][2][n-1][i]-q_1[i][0]))
                lmbd=max(abs(v_sub[index_][3][n-1][i]),abs(v_2[i][0]))
                r_f_2[i]=0.5*(n1*(fv_sub[index_][3][n-1][i]+q_2[i][0]*v_2[i][0])-lmbd*(q_s[index_][3][n-1][i]-q_2[i][0]))
             r_flux[E_no][0]=array((MS1_2*(matrix(r_f_1).T)+MS2_2*(matrix(r_f_2).T)).T)
          else:#Element is not marked for refinement
             for i in range(n):
                lamb=max(abs(v[E_no][0][i]),abs(v[boundary[E_no][0]][n-1][i]))
                r_flux[E_no][0][i]=0.5*(n1*(fv[E_no][0][i]+fv[boundary[E_no][0]][n-1][i])-lamb*(q_[boundary[E_no][0]][n-1][i]-q_[E_no][0][i]))
          if status[boundary[E_no][1]]==1:#cheking whether right element is marked for refinemen or not
             index_=numpy.where(sub_n==boundary[E_no][1])[0][0]
             u_1,u_2=array(MS1*(numpy.matrix(u[E_no,:,n-1]).T)),array(MS2*(numpy.matrix(u[E_no,:,n-1]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[E_no,:,n-1]).T)),array(MS2*(numpy.matrix(q_[E_no,:,n-1]).T))
             for i in range(n):
                lmbd=max(abs(u_sub[index_][0][i][0]),abs(u_1[i][0]))
                r_f_1[i]=0.5*(n2*(fu_sub[index_][0][i][0]+q_1[i][0]*u_1[i][0])-lmbd*(q_s[index_][0][i][0]-q_1[i][0]))
                lmbd=max(abs(u_sub[index_][2][i][0]),abs(u_2[i][0]))
                r_f_2[i]=0.5*(n2*(fu_sub[index_][2][i][0]+q_2[i][0]*u_2[i][0])-lmbd*(q_s[index_][2][i][0]-q_2[i][0]))
             r_flux[E_no][1]=array((MS1_2*(matrix(r_f_1).T)+MS2_2*(matrix(r_f_2).T)).T)
          else:#Element is not marked for refinement
             for i in range(n):
                lamb=max(abs(u[E_no][i][n-1]),abs(u[boundary[E_no][1]][i][0]))
                r_flux[E_no][1][i]=0.5*(n2*(fu[E_no][i][n-1]+fu[boundary[E_no][1]][i][0])-lamb*(q_[boundary[E_no][1]][i][0]-q_[E_no][i][n-1]))
          if (status[boundary[E_no][2]]==1):#cheking whether top element is marked for refinemen or not
             index_=numpy.where(sub_n==boundary[E_no][2])[0][0]
             v_1,v_2=array(MS1*(numpy.matrix(v[E_no,n-1,:]).T)),array(MS2*(numpy.matrix(v[E_no,n-1,:]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[E_no,n-1,:]).T)),array(MS2*(numpy.matrix(q_[E_no,n-1,:]).T))
             for i in range(n):
                lmbd=max(abs(v_sub[index_][0][0][i]),abs(v_1[i][0]))
                r_f_1[i]=0.5*(n3*(fv_sub[index_][0][0][i]+q_1[i][0]*v_1[i][0])-lmbd*(q_s[index_][0][0][i]-q_1[i][0]))
                lmbd=max(abs(v_sub[index_][1][0][i]),abs(v_2[i][0]))
                r_f_2[i]=0.5*(n3*(fv_sub[index_][1][0][i]+q_2[i][0]*v_2[i][0])-lmbd*(q_s[index_][1][0][i]-q_2[i][0]))
             r_flux[E_no][2]=array((MS1_2*(matrix(r_f_1).T)+MS2_2*(matrix(r_f_2).T)).T)
          else:#Element is not marked for refinement
             for i in range(n):
                lamb=max(abs(v[E_no][n-1][i]),abs(v[boundary[E_no][2]][0][i]))
                r_flux[E_no][2][i]=0.5*(n3*(fv[E_no][n-1][i]+fv[boundary[E_no][2]][0][i])-lamb*(q_[boundary[E_no][2]][0][i]-q_[E_no][n-1][i]))
          if status[boundary[E_no][3]]==1:#cheking whether left element is marked for refinemen or not
             index_=numpy.where(sub_n==boundary[E_no][3])[0][0]
             u_1,u_2=array(MS1*(numpy.matrix(u[E_no,:,0]).T)),array(MS2*(numpy.matrix(u[E_no,:,0]).T))
             q_1,q_2=array(MS1*(numpy.matrix(q_[E_no,:,0]).T)),array(MS2*(numpy.matrix(q_[E_no,:,0]).T))
             for i in range(n):
                lmbd=max(abs(u_sub[index_][1][i][n-1]),abs(u_1[i][0]))
                r_f_1[i]=0.5*(n4*(fu_sub[index_][1][i][n-1]+q_1[i][0]*u_1[i][0])-lmbd*(q_s[index_][1][i][n-1]-q_1[i][0]))
                lmbd=max(abs(u_sub[index_][3][i][n-1]),abs(u_2[i][0]))
                r_f_2[i]=0.5*(n4*(fu_sub[index_][3][i][n-1]+q_2[i][0]*u_2[i][0])-lmbd*(q_s[index_][3][i][n-1]-q_2[i][0]))
             r_flux[E_no][3]=array((MS1_2*(matrix(r_f_1).T)+MS2_2*(matrix(r_f_2).T)).T)
          else: #Element is not marked for refinement
             for i in range(n):
                lamb=max(abs(u[E_no][i][0]),abs(u[boundary[E_no][3]][i][n-1]))
                r_flux[E_no][3][i]=0.5*(n4*(fu[E_no][i][0]+fu[boundary[E_no][3]][i][n-1])-lamb*(q_[boundary[E_no][3]][i][n-1]-q_[E_no][i][0]))
    return r_flux,r_flux_sub
#########################################################################
#-----------------------------------------------------------------------#
########################---RK2 Solver---#################################
#-----------------------------------------------------------------------#
#########################################################################
def RK2_Solver(fu_e,fv_e,u_e,v_e,R_Flux_e,n,Qn,RDx,RDy,RF1,RF2,RF3,RF4):
    "function"
    import math
    Flux1=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    Flux2=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    Flux3=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    Flux4=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    fu_n=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    fv_n=matrix([[0.0 for col in range(1)]for row in range(Qn)])
    RHS=array([[0.0 for col in range(n)]for row in range(n)])
    j=0
    for i in range(n):#storing normal fluxes of each edge
       Flux1[i,0]=R_Flux_e[0][i]
       Flux3[Qn-1-i,0]=R_Flux_e[2][n-1-i]
       Flux2[n-1+j,0]=R_Flux_e[1][i]
       Flux4[j,0]=R_Flux_e[3][i]
       j=j+n
    fu_n,fv_n=matrix(fu_e).reshape(Qn,1),matrix(fv_e).reshape(Qn,1)
    RHS=array(((RDx*fu_n-(RF2*Flux2+RF4*Flux4)+RDy*fv_n-(RF1*Flux1+RF3*Flux3))).reshape(n,n))
    return RHS
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
print '-------------PEAK STATIC REFINEMENT---------------- \n'
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
print'[2]Inexact Integration\n'
Option=int(input('Enter your option for integration:'))
import time
start_time=time.time()
if Option==1:
    Q=(n+1)
else:
    Q=n
#CALCULATION OF DATA POINTS USING legendre_gauss_lobatto FUNCTION
[zcal,wcal]=legendre_gauss_lobatto(n);
[z,w]=legendre_gauss_lobatto(Q);
[M,Dx,Dy,F1,F2,F3,F4,RDx,RDy,RF1,RF2,RF3,RF4]=mass_diff_flux(zcal,wcal,z,w,N,Nel,Q);
# Global coordinates x,y calculation
[x,y,x_l,y_l]= global_xy(-1.0,-1.0,dx,n,Nel,Nopx,Nelxy);
Rev_T=1#2*math.pi # Time for 1 revolution of gaussian
dt=1e-3 #Time step value
T=int(Rev_T/dt) # no. of time steps required
#Initial Exact solution 
[qi,u,v]=exact(x,y,n,Nel,Nop,Nopx,Nelxy,0.0);
#exact Solution after given time
q_t=exact(x,y,n,Nel,Nop,Nopx,Nelxy,Rev_T);
fu_i=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
fv_i=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
fu_i,fv_i=qi*u,qi*v #Initial fluxes
boundary_e=array([[0 for side in range(4)]for ele in range(Nelxy)])
boundary_e=boundary_e_ele_num(Nel,Nelxy);
q=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
Rq=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn,flag=qi,0
#Calculation of projection matrices for both 1D and 2D projections
[MS1,MS2,MS1_2,MS2_2,Nopx_m,Nop_m,S1,S2,S1_20,S2_2,Mo,MM,SS1,SS2,SS3,SS4]=projection_matrices(z,w,zcal,x,y,Nel,n,Q);
#------------------------------------------------------------------------------------------#
minx=abs(x_l[0][1]-x_l[0][0])# Minimum ds
umax=2**0.5 #maximum velocity
max_dt=minx/(4*umax)#Maximum time step can be used
print 'Maximum Time Step:',max_dt
peak_index=array([0 for i in range(T)])
xc_n=array([0.0 for i in range(T)])
yc_n=array([0.0 for i in range(T)])
status=array([0 for i in range(Nelxy)])
xc,yc,T_step=-0.5,0.0,0.0
e_max=array([0.0 for i in range(T)])
#------------------------------------------------------------------------------------------#
fu_r,fv_r=fu_i,fv_i
print'Time step used for Rk2 integration:',dt
#DETERMINING PEAK POSITION AT EACH TIME STEP
for Time in range(T):# To find the location of peak for each timestep
   T_step=T_step+dt
   xc_n[Time]=-xc*math.cos(T_step)-yc*math.sin(T_step)
   yc_n[Time]=xc*math.sin(T_step)-yc*math.cos(T_step)
   for i in range(Nel):
      if ((x_l[i][0]<=xc_n[Time]) and (xc_n[Time]<=x_l[i][n-1])):
         Co_ELM=i
      if ((y_l[i][0]<=yc_n[Time]) and (yc_n[Time]<=y_l[i][n-1])):
         Ro_ELM=i
   peak_index[Time]=Ro_ELM*Nel+Co_ELM
   #peak_index[Time]=numpy.where(qn==qn.max())[0][0] # Saving element number of peak
   status[peak_index[Time]]=1
######################################################################
#--------------PEAK REFINEMENT CODE STARTSFROM HERE------------------#
#THE FOLLOWING ALGORITM REFINE ALL THE ELEMENTS WHICH CONTAIN PEAK AT ANY TIME.
n_sub,j=0,0
for i in range(Nelxy):
   if status[i]==1:
      n_sub=n_sub+1#no.of elements requiring refinement
sub_n=array([0 for i in range(n_sub)])# saving index of refined element
for i in range(Nelxy):
   if status[i]==1:
      sub_n[j],j=i,j+1
a_x=array([0.0 for i in range(n_sub)])
a_y=array([0.0 for i in range(n_sub)])
qi_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
qt_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
q_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
qn_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
fui_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
fvi_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
Rq_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
u_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
v_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
x_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
y_sub=array([[[[0.0 for col in range(n)]for row in range(n)]for ele in range(4)]for sub in range(n_sub)])
x_lsub=array([[[0.0 for col in range(n)]for ele in range(2)]for sub in range(n_sub)])
y_lsub=array([[[0.0 for col in range(n)]for ele in range(2)]for sub in range(n_sub)])
q=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
qn=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
Rq=array([[[0.0 for col in range(n)] for row in range(n)] for ele in range(Nelxy)])
for i in range(n_sub):
   a_x[i],a_y[i]=x[sub_n[i]][0][0],y[sub_n[i]][0][0]
   [x_sub[i],y_sub[i],x_lsub[i],y_lsub[i]]= global_xy(a_x[i],a_y[i],dx/2,n,2,Nopx_m,4);
   [qi_sub[i],u_sub[i],v_sub[i]]=exact(x_sub[i],y_sub[i],n,2,Nop_m,Nopx_m,4,0.0);
   qt_sub[i]=exact(x_sub[i],y_sub[i],n,2,Nop_m,Nopx_m,4,Rev_T);
   fui_sub[i],fvi_sub[i]=qi_sub[i]*u_sub[i],qi_sub[i]*v_sub[i]
fu_r,fv_r,qn,fur_sub,fvr_sub,qn_sub=fu_i,fv_i,qi,fui_sub,fvi_sub,qi_sub
#[r_flux,r_flux_sub]=normal_flux(q n_sub,fui_sub,fvi_sub,u_sub,v_sub,qn,u,v,fu_r,fv_r,Nelxy,n,status,boundary_e,sub_n,n_sub,MS1,MS2,MS1_2,MS2_2);
for Time in range(T):
   [r_flux,r_flux_sub]=normal_flux(qn_sub,fur_sub,fvr_sub,u_sub,v_sub,qn,u,v,fu_r,fv_r,Nelxy,n,status,boundary_e,sub_n,n_sub,MS1,MS2,MS1_2,MS2_2);
   for ELM in range(Nelxy): # RK2 1st step
      if status[ELM]==0:
         Rq[ELM]=RK2_Solver(fu_r[ELM],fv_r[ELM],u[ELM],v[ELM],r_flux[ELM],n,Qn,RDx,RDy,RF1,RF2,RF3,RF4);
         q[ELM]=qn[ELM]+(0.5*dt*Rq[ELM])
         fu_r[ELM],fv_r[ELM]=q[ELM]*u[ELM],q[ELM]*v[ELM]
      else:
         index=numpy.where(sub_n==ELM)[0][0]
         for j in range(4):
            Rq_sub[index][j]=RK2_Solver(fur_sub[index][j],fvr_sub[index][j],u_sub[index][j],v_sub[index][j],r_flux_sub[index][j],n,Qn,RDx,RDy,RF1,RF2,RF3,RF4);
            q_sub[index][j]=qn_sub[index][j]+(0.5*dt*2*Rq_sub[index][j])
            fur_sub[index][j],fvr_sub[index][j]=q_sub[index][j]*u_sub[index][j],q_sub[index][j]*v_sub[index][j]
   [r_flux,r_flux_sub]=normal_flux(q_sub,fur_sub,fvr_sub,u_sub,v_sub,q,u,v,fu_r,fv_r,Nelxy,n,status,boundary_e,sub_n,n_sub,MS1,MS2,MS1_2,MS2_2);
   for ELM in range(Nelxy): # RK2 2nd step
      if status[ELM]==0:
         Rq[ELM]=RK2_Solver(fu_r[ELM],fv_r[ELM],u[ELM],v[ELM],r_flux[ELM],n,Qn,RDx,RDy,RF1,RF2,RF3,RF4);
         qn[ELM]=qn[ELM]+(dt*Rq[ELM])
         fu_r[ELM],fv_r[ELM]=qn[ELM]*u[ELM],qn[ELM]*v[ELM]
      else:
         index=numpy.where(sub_n==ELM)[0][0]
         for j in range(4):
            Rq_sub[index][j]=RK2_Solver(fur_sub[index][j],fvr_sub[index][j],u_sub[index][j],v_sub[index][j],r_flux_sub[index][j],n,Qn,RDx,RDy,RF1,RF2,RF3,RF4);
            qn_sub[index][j]=qn_sub[index][j]+(dt*2*Rq_sub[index][j])
            fur_sub[index][j],fvr_sub[index][j]=qn_sub[index][j]*u_sub[index][j],qn_sub[index][j]*v_sub[index][j]
WCT=time.time()-start_time
print'Wall Clock Time:',WCT
##########Calculation of Error norms###############
SumE2,SumA2=0.0,0.0
for ELM in range(Nelxy):
   if status[ELM]==0:
      for i in range(n):
         for j in range(n):
            SumE2=SumE2+(qn[ELM][i][j]-q_t[ELM][i][j])**2
            SumA2=SumA2+q_t[ELM][i][j]**2
   else:
      index=numpy.where(sub_n==ELM)[0][0]
      for i in range(4):
         for j in range(n):
            for k in range(n):
               SumE2=SumE2+(qn_sub[index][i][j][k]-qt_sub[index][i][j][k])**2
               SumA2=SumA2+qt_sub[index][i][j][k]**2
L2=math.log10(SumE2/SumA2)*0.5
for i in range(n_sub):
    qn[sub_n[i]]=None
    #q_t[sub_n[i]]=None
q_numer=array([[0.0 for row in range(Nopx)]for col in range(Nopx)])
q_numer=Reshape_(qn,n,Nel,Nelxy,Nopx);
x_l,y_l=x_l.reshape(Nopx,1),y_l.reshape(Nopx,1)
xn,yn=numpy.meshgrid(x_l,y_l)
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab as py
fig = plt.figure()
ax = fig.gca(projection='3d')#ax =Axes3D(fig)#ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(xn,yn,q_numer,rstride=1,cstride=1,cmap=cm.RdBu)
for i in range(n_sub):
    x_p,y_p=x_lsub[i].reshape(2*n,1),y_lsub[i].reshape(2*n,1)
    xnp,ynp=numpy.meshgrid(x_p,y_p)
    q_num=Reshape_(qn_sub[i],n,2,4,2*n);
    surf=ax.plot_surface(xnp,ynp,q_num,rstride=1,cstride=1,cmap=cm.RdBu)
py.title('q_numerical(t)(x,y)')
py.xlabel('x')
py.ylabel('y')
plt.show()
"""
fig = plt.figure()
ax = fig.gca(projection='3d')#ax =Axes3D(fig)
qtplot=ax.plot_surface(xn,yn,q_anlyt,rstride=1,cstride=1,cmap=cm.RdBu)
if flag==1:
   qtplot=ax.plot_surface(xn_sub,yn_sub,q_anlsub,rstride=1,cstride=1,cmap=cm.RdBu)
py.title('q_analytical(t)(x,y)')
py.xlabel('x')
py.ylabel('y')
plt.show()"""
#################################################################
#---------------------END OF THE CODE---------------------------#
#################################################################
