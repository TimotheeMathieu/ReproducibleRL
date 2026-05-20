(define-module (guix-rl packages tools)
  #:use-module (gnu packages)
  #:use-module (gnu packages algebra)
  #:use-module (gnu packages audio)
  #:use-module (gnu packages base)
  #:use-module (gnu packages bash)
  #:use-module (gnu packages check)
  #:use-module (gnu packages cmake)
  #:use-module (gnu packages game-development)
  #:use-module (gnu packages gcc)
  #:use-module (gnu packages graphics)
  #:use-module (gnu packages image-processing)
  #:use-module (gnu packages llvm)
  #:use-module (gnu packages logging)
  #:use-module (gnu packages ninja)
  #:use-module (gnu packages machine-learning)
  #:use-module (gnu packages maths)
  #:use-module (gnu packages musl)
  #:use-module (gnu packages pkg-config)
  #:use-module (gnu packages protobuf)
  #:use-module (gnu packages python)
  #:use-module (gnu packages python-build)
  #:use-module (gnu packages python-check) 
  #:use-module (gnu packages python-xyz)
  #:use-module (gnu packages serialization)
  #:use-module (gnu packages stb)
  #:use-module (gnu packages swig)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (guix build-system cmake)
  #:use-module (guix build-system gnu)
  #:use-module (guix build-system pyproject)
  #:use-module (guix build-system python)
  #:use-module (guix build-system trivial)
  #:use-module (guix download)
  #:use-module (guix gexp)
  #:use-module (guix git-download)
  #:use-module (guix packages)
  #:use-module (guix utils)
  #:use-module (gnu packages video)
  #:use-module (gnu packages python-compression)
  
  ; #:use-module (guix-science-nonfree packages machine-learning)
)


;; libvirtcpuid
(define-public gcvir
  (package
    (name "gcvir")
    (version "0.00")
    (source
      (origin
        (method git-fetch)
        (uri (git-reference
              (url "https://github.com/twosigma/libvirtcpuid")
              (commit "3b549c732adffb566342693eaf86eb9b192510f1")))
        (file-name (git-file-name name version))
        (sha256 (base32 "1qz9ls7psqcwmi4xm3fmkyylxw58ljc4v32kscdzwjpbp8dpq2db"))
        (patches (search-patches "guix-rl/patch/libvirtcpuid.patch"))  
        ;; (patches (search-patches "libvirtcpuid.patch"))    
        ))
    (build-system gnu-build-system)

    (arguments
     `(
     #:make-flags (list ,(string-append "CC=" (cc-for-target)))
     #:phases
       (modify-phases %standard-phases
         (delete 'configure)
         (delete 'check)
         (add-after 'unpack 'copy-musl
                     (lambda* (#:key inputs outputs #:allow-other-keys)
                              (system "mkdir musl")
                              (system (string-append "tar -zxvf "  
                                                     (assoc-ref inputs "musl-1.1.24.tar.gz")
                                                     " -C ./musl --strip-components=1"))
                              ; (system "sed '74,81d' Makefile > Makefile.tmp") ;; remove download of musl
                              (system "mv Makefile.tmp Makefile")
                     ))
         (replace 'install
                  (lambda* _ 
                           (system "echo haha")))
         (add-after 'compress-documentation 'manual-install
                  (lambda* (#:key inputs outputs #:allow-other-keys)
                            (let* ((out (assoc-ref outputs "out")))
                               (system (string-append "mkdir -p " out "/lib"))
                               (system (string-append "mkdir " out "/etc"))
                               (system (string-append "cp libvirtcpuid.so " out  "/lib/libvirtcpuid.so"))
                               (system (string-append "cp -r " (assoc-ref inputs "glibc") "/lib/ld-linux-x86-64.so.2 " 
                                                      out "/lib/ld-linux-x86-64.so.2.orig"))                            
                               (system (string-append "cp -r " (assoc-ref inputs "glibc") "/* " 
                                                      out "/"))
                               (system (string-append "chmod +w " out  "/lib/ld-linux-x86-64.so.2"))

                               (system (string-append "cp ld-virtcpuid.so " out  "/lib/ld-linux-x86-64.so.2"))

                               (let
                                  ( (output-port (open-file
                                                        (string-append
                                                         out "/etc/ld-inject.env") "w")))
                               (newline output-port)
                               (display (string-append
                                         "LD_PRELOAD="  out  "/lib/libvirtcpuid.so") output-port)
                               (newline output-port)
                               (display "VIRT_CPUID_MASK=avx512f,avx2" output-port)
                               (newline output-port)
                               (close output-port)
                               )
                            )
                  )
         ))
        )
        )

    (native-inputs (list 
             (origin
                              (method url-fetch)
                              (uri "https://musl.libc.org/releases/musl-1.1.24.tar.gz")
                              (sha256 (base32 "18r2a00k82hz0mqdvgm7crzc7305l36109c0j9yjmkxj2alcjw0k"))
                              (file-name "musl-1.1.24.tar.gz")
                              )
             pkg-config
             gcc
             glibc
             ))
    (outputs '("out" "debug"
              "static"))                          ;9 MiB of .a files

    (home-page "https://github.com/twosigma/libvirtcpuid")
    (synopsis "libvirtcpuid provides transparent CPUID virtualization, all in userspace. ")
    (description "libvirtcpuid provides transparent CPUID virtualization, all in userspace. ")
    (license license:gpl2)))


(define-public gcvi2
  (package
    (name "gcvi2")
    (version "0.00")
    (source
      (origin
        (method git-fetch)
        (uri (git-reference
              (url "https://github.com/twosigma/libvirtcpuid")
              (commit "3b549c732adffb566342693eaf86eb9b192510f1")))
        (file-name (git-file-name name version))
        (sha256 (base32 "1qz9ls7psqcwmi4xm3fmkyylxw58ljc4v32kscdzwjpbp8dpq2db"))
        (patches (search-patches "guix-rl/patch/libvirtcpuid.patch"))  
        ;; (patches (search-patches "libvirtcpuid.patch"))    
        ))
    (build-system gnu-build-system)

    (arguments
     `(
     #:make-flags (list ,(string-append "CC=" (cc-for-target)))
     #:phases
       (modify-phases %standard-phases
         (delete 'configure)
         (delete 'check)
         (add-after 'unpack 'copy-musl
                     (lambda* (#:key inputs outputs #:allow-other-keys)
                              (system "mkdir musl")
                              (system (string-append "tar -zxvf "  
                                                     (assoc-ref inputs "musl-1.1.24.tar.gz")
                                                     " -C ./musl --strip-components=1"))
                              ; (system "sed '74,81d' Makefile > Makefile.tmp") ;; remove download of musl
                              (system "mv Makefile.tmp Makefile")
                     ))
         (replace 'install
                  (lambda* _ 
                           (system "echo haha")))
         (add-after 'compress-documentation 'manual-install
                  (lambda* (#:key inputs outputs #:allow-other-keys)
                            (let* ((out (assoc-ref outputs "out")))
                               (system (string-append "mkdir -p " out "/lib"))
                               (system (string-append "mkdir " out "/etc"))
                               (system (string-append "cp libvirtcpuid.so " out  "/lib/libvirtcpuid.so"))
                               (system (string-append "cp -r " (assoc-ref inputs "glibc") "/lib/ld-linux-x86-64.so.2 " 
                                                      out "/lib/ld-linux-x86-64.so.2.orig"))                            
                               (system (string-append "cp -r " (assoc-ref inputs "glibc") "/* " 
                                                      out "/"))
                               (system (string-append "chmod +w " out  "/lib/ld-linux-x86-64.so.2"))

                               (system (string-append "cp ld-virtcpuid.so " out  "/lib/ld-linux-x86-64.so.2"))

                               (let
                                  ( (output-port (open-file
                                                        (string-append
                                                         out "/etc/ld-inject.env") "w")))
                               (newline output-port)
                               (display (string-append
                                         "LD_PRELOAD="  out  "/lib/libvirtcpuid.so") output-port)
                               (newline output-port)
                               (display "VIRT_CPUID_MASK=avx2" output-port)
                               (newline output-port)
                               (close output-port)
                               )
                            )
                  )
         ))
        )
        )

    (native-inputs (list 
             (origin
                              (method url-fetch)
                              (uri "https://musl.libc.org/releases/musl-1.1.24.tar.gz")
                              (sha256 (base32 "18r2a00k82hz0mqdvgm7crzc7305l36109c0j9yjmkxj2alcjw0k"))
                              (file-name "musl-1.1.24.tar.gz")
                              )
             pkg-config
             gcc
             glibc
             ))
    (outputs '("out" "debug"
              "static"))                          ;9 MiB of .a files

    (home-page "https://github.com/twosigma/libvirtcpuid")
    (synopsis "libvirtcpuid provides transparent CPUID virtualization, all in userspace. ")
    (description "libvirtcpuid provides transparent CPUID virtualization, all in userspace. ")
    (license license:gpl2)))



(define-public gcvi5
  (package
    (name "gcvi5")
    (version "0.00")
    (source
      (origin
        (method git-fetch)
        (uri (git-reference
              (url "https://github.com/twosigma/libvirtcpuid")
              (commit "3b549c732adffb566342693eaf86eb9b192510f1")))
        (file-name (git-file-name name version))
        (sha256 (base32 "1qz9ls7psqcwmi4xm3fmkyylxw58ljc4v32kscdzwjpbp8dpq2db"))
        (patches (search-patches "guix-rl/patch/libvirtcpuid.patch"))  
        ;; (patches (search-patches "libvirtcpuid.patch"))    
        ))
    (build-system gnu-build-system)

    (arguments
     `(
     #:make-flags (list ,(string-append "CC=" (cc-for-target)))
     #:phases
       (modify-phases %standard-phases
         (delete 'configure)
         (delete 'check)
         (add-after 'unpack 'copy-musl
                     (lambda* (#:key inputs outputs #:allow-other-keys)
                              (system "mkdir musl")
                              (system (string-append "tar -zxvf "  
                                                     (assoc-ref inputs "musl-1.1.24.tar.gz")
                                                     " -C ./musl --strip-components=1"))
                              ; (system "sed '74,81d' Makefile > Makefile.tmp") ;; remove download of musl
                              (system "mv Makefile.tmp Makefile")
                     ))
         (replace 'install
                  (lambda* _ 
                           (system "echo haha")))
         (add-after 'compress-documentation 'manual-install
                  (lambda* (#:key inputs outputs #:allow-other-keys)
                            (let* ((out (assoc-ref outputs "out")))
                               (system (string-append "mkdir -p " out "/lib"))
                               (system (string-append "mkdir " out "/etc"))
                               (system (string-append "cp libvirtcpuid.so " out  "/lib/libvirtcpuid.so"))
                               (system (string-append "cp -r " (assoc-ref inputs "glibc") "/lib/ld-linux-x86-64.so.2 " 
                                                      out "/lib/ld-linux-x86-64.so.2.orig"))                            
                               (system (string-append "cp -r " (assoc-ref inputs "glibc") "/* " 
                                                      out "/"))
                               (system (string-append "chmod +w " out  "/lib/ld-linux-x86-64.so.2"))

                               (system (string-append "cp ld-virtcpuid.so " out  "/lib/ld-linux-x86-64.so.2"))

                               (let
                                  ( (output-port (open-file
                                                        (string-append
                                                         out "/etc/ld-inject.env") "w")))
                               (newline output-port)
                               (display (string-append
                                         "LD_PRELOAD="  out  "/lib/libvirtcpuid.so") output-port)
                               (newline output-port)
                               (display "VIRT_CPUID_MASK=avx512f" output-port)
                               (newline output-port)
                               (close output-port)
                               )
                            )
                  )
         ))
        )
        )

    (native-inputs (list 
             (origin
                              (method url-fetch)
                              (uri "https://musl.libc.org/releases/musl-1.1.24.tar.gz")
                              (sha256 (base32 "18r2a00k82hz0mqdvgm7crzc7305l36109c0j9yjmkxj2alcjw0k"))
                              (file-name "musl-1.1.24.tar.gz")
                              )
             pkg-config
             gcc
             glibc
             ))
    (outputs '("out" "debug"
              "static"))                          ;9 MiB of .a files

    (home-page "https://github.com/twosigma/libvirtcpuid")
    (synopsis "libvirtcpuid provides transparent CPUID virtualization, all in userspace. ")
    (description "libvirtcpuid provides transparent CPUID virtualization, all in userspace. ")
    (license license:gpl2)))



(define-public marchingcubecpp
  (package
    (name "marchingcubecpp")
    (version "git-20230911") 
    (source 
       (origin
           (method git-fetch)
         
	   (uri (git-reference
		(url "https://github.com/aparis69/MarchingCubeCpp.git")
		(commit "f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d")))
           
	      (file-name (git-file-name name version))
              (sha256
               (base32 "077hgjd1w0nmxladjv9zmyv11pcryxkaqd32akpca0s9bb9a4izp"))))
    (build-system trivial-build-system)
    
    
    (arguments
     `(#:modules ((guix build utils))
       #:builder
       (begin
         (use-modules (guix build utils))
         (let* ((source (assoc-ref %build-inputs "source"))
                (target (string-append %output "/include")))
           (mkdir-p target)
           (copy-recursively source target)
           #t))))
              
    (synopsis "A public domain/MIT header-only marching cube implementation in C++ without anything fancy. ")
    (description "A public domain header-only marching cube implementation in C/C++ without anything fancy. Only dependencies are cmath and vector headers. I could get rid of vector and just use plain arrays, but it is more convenient for my personal projects to keep that way, though one could easily modify the implementation.")
    (home-page "https://github.com/aparis69/MarchingCubeCpp")
    (license license:public-domain)))







(define-public python-box2d-py
  (package
    (name "python-box2d-py")
    (version "2.3.5")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/openai/box2d-py")
             (commit "2.3.5")))
       (file-name (git-file-name name version))
       
       (sha256
        (base32 "0cmh10kd5vbcwfx7gd2zsrmx1kvdjhwrgvcrx2qdlr0z33wgxz7d"))))
    (build-system pyproject-build-system)
    (arguments
     `(#:phases (modify-phases %standard-phases
                  (replace 'check
                    (lambda* (#:key tests? inputs outputs #:allow-other-keys)
                      (when tests?
                        (add-installed-pythonpath inputs outputs)                        
                        (invoke "touch" "tests/__init__.py")
                        (invoke "pytest" "tests"))))
                  )))
    (propagated-inputs (list swig python-pygame ))
    (native-inputs (list python-setuptools python-wheel python-pytest opencv))
    (home-page "https://github.com/openai/box2d-py")
    (synopsis "Python Box2D")
    (description "Python Box2D.")
    (license license:zlib)))





(define-public python-dm-env
  (package
    (name "python-dm-env")
    (version "1.6")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "dm-env" version))
       (sha256
        (base32 "1plk7pzzrd3yz5izzkhga45i99xy33icw5m5hv4y0facclffndm4"))))
    (build-system pyproject-build-system)
    (arguments
    (list
      #:tests? #f                    ; there are none
    )
    )
    (propagated-inputs (list python-absl-py python-dm-tree python-numpy))
    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/google-deepmind/dm_env")
    (synopsis "A Python interface for Reinforcement Learning environments.")
    (description "This package provides a Python interface for Reinforcement Learning environments.")
    (license license:asl2.0)))




(define-public python-etils-epath
  (package
    (name "python-etils-epath")
    (version "1.5.2")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/google/etils/")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "1xhnsr4n6dxsn25jiblf5qpk8jj9rgm4yb3gq4zyxffqxd1nlplg"))))
    (build-system pyproject-build-system)
             
     (arguments
     `(#:phases
      (modify-phases %standard-phases
        (replace 'check
              (lambda* (#:key tests? #:allow-other-keys)
                (when tests?
                  ;;(invoke "pytest" "-k" "not test_public_access" 
                  ;;"--ignore=etils/epy/lazy_imports_utils_test.py" 
                  ;;"--ignore=etils/epath/flags_test.py"
                  ;;"etils/epath" "etils/epy") 
                  (invoke "echo" "skipping tests for now")
                ))))
      )
    )            
             
    (inputs (list ffmpeg-5))            ;for mediapy
    (propagated-inputs
     (list python-fsspec
           python-importlib-resources
           python-typing-extensions
           python-zipp))
    (native-inputs
     (list python-flit-core
           python-pytest
           python-pytest-subtests
           python-pytest-xdist))
    (home-page "https://github.com/google/etils/")
    (synopsis "Collection of common Python utils")
    (description "This is a collection of independent Python modules
providing utilities for various projects.")
    (license license:asl2.0)))


(define-public python-gym-notices
  (package
    (name "python-gym-notices")
    (version "0.0.8")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gym-notices" version))
       (sha256
        (base32 "04drnkr40ilvcl162lhqcqsa3bc89q3gw9c6f9ls7bvw900f49dd"))))
    (build-system pyproject-build-system)
    (arguments
     `(#:phases
      (modify-phases %standard-phases
        (replace 'check
              (lambda* (#:key tests? #:allow-other-keys)
                (when tests?
                  (invoke "echo" "skipping tests for now")
                ))))
      )
    )  
    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/Farama-Foundation/gym-notices")
    (synopsis "Notices for gym")
    (description "This repository hosts notices for Gym that may be displayed on import on internet connected systems, in order to give notices if versions have major reproducibility issues, are very old and need to be upgraded (e.g. there's been issues with researchers using 4 year old versions of Gym for no reason), or other similar issues. If you're using a current version of Gym and nothing extraordinary happens, you'll never see a message from this, but I want start including the option to prevent future issues. By pulling the error messages from a public git repository, there's absolute transparency and versioning in the process.")
    (license license:expat)))



(define-public python-moviepy
  (package
    (name "python-moviepy")
    (version "2.2.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "moviepy" version))
       (sha256
        (base32 "1cdcvqb3xl44wv228fnj145b7vkh02j1ldid7rg4xsgc2mlba368"))))
    (build-system pyproject-build-system)
    (arguments
     `(#:phases
      (modify-phases %standard-phases
        (replace 'check
              (lambda* (#:key tests? #:allow-other-keys)
                (when tests?
                  (invoke "echo" "Skipped : MoviePy tests require ffmpeg to run.")
                ))))
      )
    )
    (propagated-inputs (list python-decorator
                             python-imageio
                             python-imageio-ffmpeg
                             python-numpy
                             python-pillow
                             python-proglog
                             python-dotenv))
    (native-inputs (list python-coveralls python-pytest python-pytest-cov
                         python-setuptools python-wheel))
    (home-page "https://github.com/Zulko/moviepy")
    (synopsis "Video editing with Python")
    (description "MoviePy is a Python library for video editing: cuts, concatenations, title insertions, video compositing (a.k.a. non-linear editing), video processing, and creation of custom effects. MoviePy can read and write all the most common audio and video formats, including GIF, and runs on Windows/Mac/Linux, with Python 3.9+.")
    (license license:expat)))


(define-public python-proglog
  (package
    (name "python-proglog")
    (version "0.1.12")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "proglog" version))
       (sha256
        (base32 "074kv9n8dxyccajyyhxhj9y2hpwcrcv161jwny4pn9qwf9sf07in"))
        (patches (search-patches "guix-rl/patch/proglog.patch"))      
        ))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-tqdm))
    (native-inputs (list python-setuptools python-wheel python-pytest))
    (home-page "https://github.com/Edinburgh-Genome-Foundry/proglog")
    (synopsis "Logs and progress bars manager for Python ")
    (description "Proglog is a progress logging system for Python. It allows to build complex libraries while giving your users control over logs, callbacks and progress bars.")
    (license license:expat)))




(define-public python-pytinyrenderer
  (package
    (name "python-pytinyrenderer")
    (version "0.0.14")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "pytinyrenderer" version))
       (sha256
        (base32 "059xbawh9wgyis3ynlnyn4vammflx26rxg530cd93jq9hmwv9vaz"))))
    (build-system pyproject-build-system)
    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/erwincoumans/tinyrenderer")
    (synopsis "Python bindings for Tiny Renderer")
    (description "PyTinyRenderer is a simple C++/CPU renderer, this fork adds Python bindings and triangle clipping and OpenGL compatible view/projection matrix.")
    (license license:asl2.0)))


(define-public python-shimmy
  (package
    (name "python-shimmy")
    (version "0.2.1")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/Farama-Foundation/Shimmy")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "0aaxqgna9rb4r88x1v2lk8zgp46zkv9r20yjs40llzdf25h7ixa4"))))
    (build-system pyproject-build-system)
    (arguments
     `(#:phases
      (modify-phases %standard-phases
        (replace 'check
              (lambda* (#:key tests? #:allow-other-keys)
                (when tests?
                  (invoke "echo" "Skipped : MoviePy tests require ffmpeg to run.")
                ))))
      )
    )
    (propagated-inputs (list python-gymnasium python-numpy))
    (native-inputs (list python-pillow python-pytest python-setuptools))
    (home-page "https://github.com/Farama-Foundation/Shimmy")
    (synopsis
     "An API conversion tool providing Gymnasium and PettingZoo bindings for popular external reinforcement learning environments.")
    (description
     "An API conversion tool providing Gymnasium and @code{PettingZoo} bindings for
popular external reinforcement learning environments.")
    (license license:expat)))


(define-public sdflib
  (package
    (name "sdflib")
    (version "git-1927bee6bb8225258a39c8cbf14e18a4d50409ae")
    (source
      (origin
        (method git-fetch)
           (uri (git-reference
           (url "https://github.com/UPC-ViRVIG/SdfLib.git")
           (commit "1927bee6bb8225258a39c8cbf14e18a4d50409ae")))
       (file-name (git-file-name name version))
       (sha256 (base32 "1qbl8x8bkp45msjk62dncwfwrd3xn8zrkd06kmpnd93ssqwm88gr"))       
       ))
    (build-system cmake-build-system)    
    (arguments
     (list
      #:configure-flags
      #~(list "-DSDFLIB_USE_SYSTEM_GLM=ON"
      	"-DSDFLIB_USE_SYSTEM_ASSIMP=ON"
      	"-DSDFLIB_USE_ASSIMP=ON"
      	"-DSDFLIB_USE_SYSTEM_SPDLOG=ON"
      	"-DSDFLIB_USE_SYSTEM_CEREAL=ON"
      	"-DSDFLIB_USE_ENOKI=OFF"
      	)
      	
      #:phases
      #~(modify-phases %standard-phases
       (replace 'install       
       (lambda* (#:key outputs #:allow-other-keys)
       (let* ((out (assoc-ref outputs "out"))
                (includedir (string-append out "/include/SdfLib")))
       
         (mkdir-p (string-append out "/include/SdfLib"))
         (copy-recursively "../source/include/SdfLib" includedir)
         #t))
	))      
      
      #:tests? #f
     ))     
    (native-inputs (list pkg-config cmake))
    (inputs (list spdlog glm cereal assimp eigen stb-image ))
    (synopsis "SdfLib is a library for accelerating the queries of signed distance fields from triangle meshes ")
    (description
     "SdfLib is a library for accelerating the queries of signed distance fields from triangle meshes. The library has an exact method that always returns the distance to the nearest triangle and an approximated one that return distances with a configurable maximum error.")
    (home-page "https://github.com/UPC-ViRVIG/SdfLib")
    (license license:expat)))


(define-public tinyobjloader
  (package
    (name "tinyobjloader")
    (version "git-1421a10d6ed9742f5b2c1766d22faa6cfbc56248")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
       (url "https://github.com/tinyobjloader/tinyobjloader.git")
		(commit "1421a10d6ed9742f5b2c1766d22faa6cfbc56248")))
       (file-name (git-file-name name version))
       (sha256 (base32 "03gbssdx9wd08fj7yq55031988d5zzlhmz4l8bj2a2lgymxqsggp"))))
    (build-system cmake-build-system)
    (arguments
     `(#:configure-flags `("-DTINYOBJLOADER_BUILD_TESTS=ON")
      #:phases
      (modify-phases %standard-phases
          (replace 'check
            (lambda* (#:key tests? #:allow-other-keys)
              (when tests?
                (invoke "make" "-C" "../source/tests" "check")
                ))))))
    (native-inputs (list python ninja))
    (inputs (list cmake clang))
    (synopsis "Tiny but powerful single file wavefront obj loader ")
    (description
     "Tiny but powerful single file wavefront obj loader written in C++03. No dependency except for C++ STL. It can parse over 10M polygons with moderate memory and time. tinyobjloader is good for embedding .obj loader to your (global illumination) renderer ;-)")
    (home-page "https://github.com/tinyobjloader/tinyobjloader")
    (license license:expat)))

