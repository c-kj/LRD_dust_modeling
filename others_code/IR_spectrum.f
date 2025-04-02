      PROGRAM IR_Tdust
      implicit none
      integer :: i
      real*8 :: rho0, r, Luv, Td, gamma, rho, r_in, r_out, T_sub,
     &     dr, NH
      real*8 :: pi, sigma_SB, pc

      pi = 3.141592d0
      sigma_SB = 5.67d-5
      pc = 3.085d18

      Luv = 1.0d46 ! erg/s
      T_sub = 1.2d3 ! K
      rho0 = 1.0d3
      gamma = 0.0d0
      
      r_in = dsqrt(Luv/(16.0d0*pi*sigma_SB)/(T_sub**4.0d0))

      dr = r_in*1.0d-2
      NH = 0.0d0
      r = r_in
      do i=1,1000
         r = r + dr
         rho = rho0 / (r/r_in)**gamma
         NH = NH + rho*dr

         Td = 1.d3*(Luv/1.0d46/(r/pc)**2.0d0)**0.1785d0
     &        *dexp(-NH/1.6d22/5.6d0)
!         Td = 1.65d3*(Luv/1.0d46/(r/pc)**2.0d0)**0.1785d0
!     &        *dexp(-NH/1.3d22/5.6d0)
         
         dr = 1.01d0*dr
!         if(NH>1.d23) exit
!         print*, r/pc, (r-r_in)/pc, NH/1.d22, Td
         write(10,*)  r/pc, NH/1.d22, Td
      enddo



      END
      
