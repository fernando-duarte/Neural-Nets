using Pkg
Pkg.add("SpecialFunctions")
Pkg.build("SpecialFunctions")
Pkg.add("FFTW")
Pkg.build("FFTW")
ENV["GRDIR"]="" ; Pkg.build("GR")
Pkg.add("PATHSolver")
Pkg.build("PATHSolver")
Pkg.add("SLEEFPirates")
Pkg.build("SLEEFPirates")
Pkg.add("IterTools")
Pkg.add("Tracker")
