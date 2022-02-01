

import gaussian, student, archimedean



import numpy as np

cop = gaussian.gaussian()

lvls = [0.01, 0.1, 0.5, 1, 1.5, 2, 3]

cop.plot_pdf(0.8, type="3d", Nsplit=40, )
cop.plot_pdf(0.8,  type="contour", Nsplit=40,  lvls=[0.01,0.1,0.4,0.8,1.3,1.6] )
cop.plot_cdf(0.8, type="3d", Nsplit=40,)
cop.plot_cdf(0.8, type="contour", Nsplit=40, lvls=np.linspace(0.1,1,10), color="black",  )
cop.plot_mpdf(0.8, margin=["gaussian", "gaussian"], type="3d", Nsplit=50)
cop.plot_mpdf(0.8, margin=["gaussian", "gaussian"], type="contour", Nsplit=50)


cop = student.student()

cop.plot_pdf([0.8,3], type="3d", Nsplit=40, )
cop.plot_pdf([0.8,3],  type="contour", Nsplit=40,  lvls=lvls )


print("galambos")
cop = archimedean.archimedean(family="galambos")

cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=100,)
cop.plot_pdf([2],  type="contour", Nsplit=100,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=100, lvls=np.linspace(0.1,1,10), color="black",  )

cop = archimedean.archimedean(family="BB1")

cop.plot_pdf([2,1], type="3d", Nsplit=100, )
cop.plot_pdf([2,1], type="3d", Nsplit=100,  )
cop.plot_cdf([2,1], type="3d", Nsplit=100,)
cop.plot_pdf([2,1],  type="contour", Nsplit=100,  lvls=lvls )
cop.plot_cdf([2,1], type="contour", Nsplit=100, lvls=np.linspace(0.1,1,10), color="black",  )


print("rgalambos")
cop = archimedean.archimedean(family="rgalambos")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_pdf([2], type="3d", Nsplit=100,)
cop.plot_cdf([2], type="3d", Nsplit=100,)
cop.plot_pdf([2],  type="contour", Nsplit=100,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=100, lvls=np.linspace(0.1,1,10), color="black",  )


print("clayton")
cop = archimedean.archimedean(family="clayton")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=100,)
cop.plot_pdf([2],  type="contour", Nsplit=100,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=100, lvls=np.linspace(0.1,1,10), color="black",  )


print("rclayton")
cop = archimedean.archimedean(family="rclayton")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=100,)
cop.plot_pdf([2],  type="contour", Nsplit=100,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=100, lvls=np.linspace(0.1,1,10), color="black",  )


print("gumbel")
cop = archimedean.archimedean(family="gumbel")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=100,)
cop.plot_pdf([2],  type="contour", Nsplit=100,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=100, lvls=np.linspace(0.1,1,10), color="black",  )


print("rgumbel")
cop = archimedean.archimedean(family="rgumbel")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=60,)
cop.plot_pdf([2],  type="contour", Nsplit=60,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=60, lvls=np.linspace(0.1,1,10), color="black",  )

print("plackett")
cop = archimedean.archimedean(family="plackett")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=60,)
cop.plot_pdf([2],  type="contour", Nsplit=60,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=60, lvls=np.linspace(0.1,1,10), color="black",  )


print("fgm")
cop = archimedean.archimedean(family="fgm")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=60,)
cop.plot_pdf([2],  type="contour", Nsplit=60,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=60, lvls=np.linspace(0.1,1,10), color="black",  )


print("frank")
cop = archimedean.archimedean(family="frank")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=60,)
cop.plot_pdf([2],  type="contour", Nsplit=60,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=60, lvls=np.linspace(0.1,1,10), color="black",  )


print("joe")
cop = archimedean.archimedean(family="joe")
cop.plot_pdf([2], type="3d", Nsplit=100, )
cop.plot_cdf([2], type="3d", Nsplit=100,)
cop.plot_pdf([2],  type="contour", Nsplit=60,  lvls=lvls )
cop.plot_cdf([2], type="contour", Nsplit=60, lvls=np.linspace(0.1,1,10), color="black",  )

