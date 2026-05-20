(define-module (guix-rl packages reinforcement-learning)
  #:use-module (gnu packages)
  #:use-module (gnu packages algebra)
  #:use-module (gnu packages base)
  #:use-module (gnu packages benchmark)
  #:use-module (gnu packages check)
  #:use-module (gnu packages cmake)
  #:use-module (gnu packages compression)
  #:use-module (gnu packages cpp)
  #:use-module (gnu packages game-development)  
  #:use-module (gnu packages gl)
  #:use-module (gnu packages image)
  #:use-module (gnu packages image-processing)  
  #:use-module (gnu packages logging)
  #:use-module (gnu packages machine-learning)
  #:use-module (gnu packages maths)
  #:use-module (gnu packages ninja)  
  #:use-module (gnu packages pkg-config)  
  #:use-module (gnu packages python)
  #:use-module (gnu packages python-build)
  #:use-module (gnu packages python-check)
  #:use-module (gnu packages python-compression)  
  #:use-module (gnu packages python-graphics)
  #:use-module (gnu packages python-science)
  #:use-module (gnu packages python-web) 
  #:use-module (gnu packages python-xyz)
  #:use-module (gnu packages sdl)
  #:use-module (gnu packages serialization)
  #:use-module (gnu packages swig)  
  #:use-module (gnu packages time)
  #:use-module (gnu packages version-control)
  #:use-module (gnu packages xml)
  #:use-module (gnu packages xorg)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (guix build utils)
  #:use-module (guix build-system cmake)
  #:use-module (guix build-system python)
  #:use-module (guix build-system pyproject)
  #:use-module (guix download)
  #:use-module (guix gexp)
  #:use-module (guix git-download)
  #:use-module (guix packages)
  #:use-module (guix utils)
  #:use-module (guix-rl packages tools)
)





(define tinyxml2-old
  (package (inherit tinyxml2)
           (version "8.0.0")
           (name "tinyxml2-old")
           (source
            (origin
             (method git-fetch)
             (uri (git-reference
                   (url "https://github.com/leethomason/tinyxml2")
                   (commit version)))
             (file-name (git-file-name name version))
             (sha256
              (base32 "0raa8r2hsagk7gjlqjwax95ib8d47ba79n91r4aws2zg8y6ssv1d"))))
           ))

(define-public python-pandas-new
  (package (inherit python-pandas)
           (version "2.3.3")
           (name "python-pandas-new")
           (source
            (origin
             (method url-fetch)
             (uri (pypi-uri "pandas" version))
             (sha256
              (base32 "0fwr9kk44n9110sw36dgh6nf6zjcz7wl7l1nlsppwzwp7gwilpp0"))))
           (build-system pyproject-build-system)
           ))


(define-public mujoco
   (package
    (name "mujoco")
    (version "3.3.1")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/google-deepmind/mujoco")
             (commit "3.3.1")))
       (file-name (git-file-name name version))
       (sha256
        (base32 "1gzsw66z9zmwyv6imrv42lcckr775wpn616wk608zfng7pxq3c6m"))
        (patches (search-patches "guix-rl/patch/disable-mujoco-fetch.patch"))
        ))
    (build-system cmake-build-system )
    (arguments
     (list
      #:configure-flags
      #~(list "-DBUILD_TEST=OFF"
              "-DBUILD_EXAMPLES=OFF"
              "-DMUJOCO_ENABLE_PYTHON=OFF" 
              "-DCMAKE_CXX_FLAGS=-DSPDLOG_FMT_EXTERNAL=ON"
              (string-append "-DLODEPNG_PATH=" (assoc-ref %build-inputs "lodepng"))
              (string-append "-DQHULL_PATH=" (assoc-ref %build-inputs "qhull"))
              (string-append "-DTINYXML2_PATH=" (assoc-ref %build-inputs "tinyxml2-old"))
              (string-append "-DLIBCCD_PATH=" (assoc-ref %build-inputs "libccd"))
              (string-append "-DABSEIL_CPP_PATH=" (assoc-ref %build-inputs "abseil-cpp"))
              (string-append "-DBENCHMARK_PATH=" (assoc-ref %build-inputs "benchmark"))
              (string-append "-DEIGEN_PATH=" (assoc-ref %build-inputs "eigen"))
              (string-append "-DMARCHINGCUBECPP_PATH=" (assoc-ref %build-inputs "marchingcubecpp"))
              (string-append "-DTINYOJLOADER_PATH=" (assoc-ref %build-inputs "tinyobjloader"))
              (string-append "-DSDFLIB_PATH=" (assoc-ref %build-inputs "sdflib"))
              (string-append "-DGTEST_PATH=" (assoc-ref %build-inputs "googletest"))
              (string-append "-DGLFW_PATH=" (assoc-ref %build-inputs "glfw"))
              (string-append "-DCMAKE_PREFIX_PATH=" (assoc-ref %build-inputs "lodepng") ":"
              (assoc-ref %build-inputs "qhull") ":"
              (assoc-ref %build-inputs "tinyxml2-old") ":"
              (assoc-ref %build-inputs "libccd") ":"
              (assoc-ref %build-inputs "abseil-cpp") ":"
              (assoc-ref %build-inputs "benchmark") ":"
              (assoc-ref %build-inputs "eigen") ":"
              (assoc-ref %build-inputs "marchingcubecpp") ":"
              (assoc-ref %build-inputs "tinyobjloader") ":" 
              (assoc-ref %build-inputs "sdflib") ":"
              (assoc-ref %build-inputs "googletest") ":"
              (assoc-ref %build-inputs "glfw") ":"
              )) 
        #:tests? #f        
                   
        )
        )                    
    (native-inputs (list pkg-config))
    (inputs (list glfw glew libxi libxmu libx11 lodepng qhull tinyxml2-old libccd abseil-cpp googletest benchmark eigen marchingcubecpp tinyobjloader sdflib glm spdlog cereal))    
    (home-page "https://github.com/google-deepmind/mujoco")
    (synopsis
     "MuJoCo physics engine (C/C++ core)")
    (description
     "MuJoCo stands for Multi-Joint dynamics with Contact. It is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas which demand fast and accurate simulation of articulated structures interacting with their environment.")
    (license license:asl2.0)))



(define-public python-ale-py
  (package
    (name "python-ale-py")
    (version "0.10.2")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/Farama-Foundation/Arcade-Learning-Environment")
             (commit "v0.10.2")))
       (file-name (git-file-name name version))
       (sha256
        (base32 "16hjl5k5bp3s50vvn7j4xdgp0ig7x17j2xwpz2n9lh98ai02ar88"))
        ))
    (build-system pyproject-build-system)
    (propagated-inputs (list (specification->package "python-numpy@1.26.4")))
    (inputs (list cmake zlib sdl2 pybind11 ninja))

    (native-inputs
    `( 
      ("autorom-roms" ,autorom-roms)
      ("python-farama-notifications" ,python-farama-notifications)
      ("python-setuptools" ,python-setuptools)
      ("python-wheel" ,python-wheel)))

    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-before 'build 'patch-rom-download
            (lambda* (#:key inputs outputs #:allow-other-keys)
              (let* ((roms-archive (assoc-ref inputs "autorom-roms"))
                     (out (assoc-ref outputs "out"))
                     (rom-dir (string-append out "/share/autorom/roms"))
                     (bin-extract-path (string-append rom-dir "/ROM/*/*.bin"))                    
                     (bin-final-path (string-append (getcwd) "/src/ale/python/roms/"))
                     (full-command (string-append "base64" " -d " roms-archive " | " "tar " "-xzf " "- " "-C " rom-dir))
                     (mv-command (string-append "mv " bin-extract-path " " bin-final-path))
                     )
                (mkdir-p rom-dir)                
                (invoke "sh" "-c" full-command)
                (invoke "sh" "-c" mv-command)
                
                #t)))
        (replace 'check
      	(lambda* (#:key tests? #:allow-other-keys)
        (when tests?
          (invoke "echo" "skipping tests for now")
        ))))
        ))

    (home-page "https://github.com/Farama-Foundation/Arcade-Learning-Environment")
    (synopsis
     "The Arcade Learning Environment (ALE) -- a platform for AI research...")
    (description
     "The Arcade Learning Environment (ALE) is a simple framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games. It is built on top of the Atari 2600 emulator Stella and separates the details of emulation from agent design. This video depicts over 50 games currently supported in the ALE.")
    (license license:gpl2)))

(define-public autorom-roms    
  (origin
    (method url-fetch)
      (uri "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64")
       (sha256
        (base32 "1php65js0dszk9a36l0mn8qsrhr4iykjn2k86vx74sj72ry7gjh2"))))


(define-public python-gym-old
  (package
    (name "python-gym-old")
    (version "0.26.2")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gym" version))
       (sha256
        (base32 "1i5w9whvxk82gly811lr53qkkc1qiamj8k0h0gr6a32gnps85n70"))))
    (build-system pyproject-build-system)
    (arguments
    (list
      #:tests? #f                    ; there are none
    )
    )
    (propagated-inputs (list python-cloudpickle python-gym-notices
                             (specification->package "python-numpy@1.26.4")))
    (native-inputs (list python-box2d-py
                         python-imageio
                         python-lz4
                         python-matplotlib
                         python-moviepy
                         python-mujoco
                        ;  python-mujoco-py
                         opencv
                         python-pygame
                         python-pytest
                         swig
                         python-setuptools
                         python-wheel
                         pkg-config
                         ))
    (home-page "https://www.gymlibrary.dev/")
    (synopsis "A toolkit for developing and comparing reinforcement learning algorithms.")
    (description "***The team that has been maintaining Gym since 2021 has moved all future development to Gymnasium, a drop in replacement for Gym (import gymnasium as gym), and Gym will not be receiving any future updates. Please switch over to Gymnasium as soon as you're able to do so.***
    Gym is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API.")
    (license license:expat)))



(define-public python-gymnasium-ale-py
  (package
    (name "python-gymnasium-ale-py")
    (version "1.1.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gymnasium" version))
       (sha256
        (base32 "0va5s8h3r5aqa11k0bf33q2q27ayg3y6mwsg8h59ab7kvsdymncb"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'unpack 'create-tests-module
            (lambda _
              (with-output-to-file "tests/__init__.py"
                (lambda _ (display ""))))))))
    (native-inputs (list python-dill python-pytest python-scipy
    			 python-gymnasium-next python-ale-py
                         python-setuptools python-wheel))
    (home-page "https://gymnasium.farama.org/environments/box2d/")
    (synopsis
     "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym) with Box2D.")
    (description
     "This package provides a standard API for reinforcement learning and a diverse
set of reference environments (formerly Gym).
These environments all involve toy games based around physics control, using box2d based physics and PyGame-based rendering. These environments were contributed back in the early days of OpenAI Gym by Oleg Klimov, and have become popular toy benchmarks ever since. All environments are highly configurable via arguments specified in each environment’s documentation.")
    (license license:expat)))




(define-public python-gymnasium-all
  (package
    (name "python-gymnasium-all")
    (version "1.1.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gymnasium" version))
       (sha256
        (base32 "0va5s8h3r5aqa11k0bf33q2q27ayg3y6mwsg8h59ab7kvsdymncb"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'unpack 'create-tests-module
            (lambda _
              (with-output-to-file "tests/__init__.py"
                (lambda _ (display ""))))))))
    (native-inputs (list python-dill python-pytest python-scipy python-box2d-py python-gymnasium-next python-mujoco python-ale-py python-setuptools python-wheel))
    (home-page #f)
    (synopsis
     "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym).")
    (description
     "This package provides a standard API for reinforcement learning and a diverse
set of reference environments (formerly Gym).")
    (license license:expat)))


(define-public python-gymnasium-box2d
  (package
    (name "python-gymnasium-box2d")
    (version "1.1.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gymnasium" version))
       (sha256
        (base32 "0va5s8h3r5aqa11k0bf33q2q27ayg3y6mwsg8h59ab7kvsdymncb"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'unpack 'create-tests-module
            (lambda _
              (with-output-to-file "tests/__init__.py"
                (lambda _ (display ""))))))))
    (native-inputs (list python-dill python-pytest python-scipy
    			 python-gymnasium-next python-box2d-py
                         python-setuptools python-wheel))
    (home-page "https://gymnasium.farama.org/environments/box2d/")
    (synopsis
     "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym) with Box2D.")
    (description
     "This package provides a standard API for reinforcement learning and a diverse
set of reference environments (formerly Gym).
These environments all involve toy games based around physics control, using box2d based physics and PyGame-based rendering. These environments were contributed back in the early days of OpenAI Gym by Oleg Klimov, and have become popular toy benchmarks ever since. All environments are highly configurable via arguments specified in each environment’s documentation.")
    (license license:expat)))



(define-public python-gymnasium-classic-control
  (package
    (name "python-gymnasium-classic-control")
    (version "1.1.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gymnasium" version))
       (sha256
        (base32 "0va5s8h3r5aqa11k0bf33q2q27ayg3y6mwsg8h59ab7kvsdymncb"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'unpack 'create-tests-module
            (lambda _
              (with-output-to-file "tests/__init__.py"
                (lambda _ (display ""))))))))
    (native-inputs (list python-dill python-pytest python-scipy
    			 python-gymnasium-next python-pygame
                         python-setuptools python-wheel))
    (home-page #f)
    (synopsis
     "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym).")
    (description
     "This package provides a standard API for reinforcement learning and a diverse
set of reference environments (formerly Gym).")
    (license license:expat)))



(define-public python-gymnasium-mujoco
  (package
    (name "python-gymnasium-mujoco")
    (version "1.1.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gymnasium" version))
       (sha256
        (base32 "0va5s8h3r5aqa11k0bf33q2q27ayg3y6mwsg8h59ab7kvsdymncb"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'unpack 'create-tests-module
            (lambda _
              (with-output-to-file "tests/__init__.py"
                (lambda _ (display ""))))))))
    (propagated-inputs (list python-dill python-pytest python-scipy
    			 python-gymnasium-next python-mujoco python-cython
                         python-setuptools python-wheel))
    (home-page #f)
    (synopsis
     "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym).")
    (description
     "This package provides a standard API for reinforcement learning and a diverse
set of reference environments (formerly Gym).")
    (license license:expat)))



(define-public python-gymnasium-next
  (package
    (name "python-gymnasium-next")
    (version "1.1.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "gymnasium" version))
       (sha256
        (base32 "0va5s8h3r5aqa11k0bf33q2q27ayg3y6mwsg8h59ab7kvsdymncb"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      #~(modify-phases %standard-phases
          (add-after 'unpack 'create-tests-module
            (lambda _
              (with-output-to-file "tests/__init__.py"
                (lambda _ (display ""))))))))
    (propagated-inputs (list python-cloudpickle python-farama-notifications
                             python-importlib-metadata (specification->package "python-numpy@1.26.4")
                             python-typing-extensions))
    (native-inputs (list python-dill python-pytest python-scipy python-imageio
                         python-setuptools python-wheel))
    (home-page #f)
    (synopsis
     "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym).")
    (description
     "This package provides a standard API for reinforcement learning and a diverse
set of reference environments (formerly Gym).")
    (license license:expat)))



(define-public python-minatar
  (package
    (name "python-minatar")
    (version "1.0.15")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "MinAtar" version))
       (sha256
        (base32 "0fikcdbjl7ymnkxsd00lmca4lazm48qr8xgllj36w2nr55xbz6r7"))))
    (build-system pyproject-build-system)

    (arguments
    (list
      #:tests? #f                    ; there are none
    )
    )

    (propagated-inputs (list python-cycler
                             python-kiwisolver
                             python-matplotlib
                             (specification->package "python-numpy@1.26.4")
                             python-pandas
                             python-pyparsing
                             python-dateutil
                             python-pytz
                             python-scipy
                             python-seaborn
                             python-six
                             python-gymnasium-next
                             ))
    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/kenjyoung/MinAtar")
    (synopsis "A miniaturized version of the Arcade Learning Environment.")
    (description
     "MinAtar is a testbed for AI agents which implements miniaturized versions of several Atari 2600 games. MinAtar is inspired by the Arcade Learning Environment (Bellemare et. al. 2013) but simplifies the games to make experimentation with the environments more accessible and efficient. Currently, MinAtar provides analogues to five Atari games which play out on a 10x10 grid. The environments provide a 10x10xn state representation, where each of the n channels correspond to a game-specific object, such as ball, paddle and brick in the game Breakout.")
    (license license:gpl3)))



(define-public python-mujoco
  (package
    (name "python-mujoco")
    (version "3.3.1")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "mujoco" version))
       (sha256
        (base32 "0a0hi0npv4gzhcrdb2z7jk5nnwi8prz6bs40ic62r0k72mfa652j"))
        (patches (search-patches "guix-rl/patch/disable-python-mujoco-fetch.patch"))
        ))
    (build-system pyproject-build-system)
        (arguments
          (list

            #:phases
            #~(modify-phases %standard-phases
                (add-before 'build 'set-env
                  (lambda* (#:key inputs #:allow-other-keys)
                    (setenv "MUJOCO_PATH" (assoc-ref inputs "mujoco"))
                    (setenv "MUJOCO_PLUGIN_PATH" (string-append (assoc-ref inputs "mujoco") "/share/mujoco/model/plugin/") )
                    (setenv "GLFW_PATH" (string-append (assoc-ref inputs "glfw")))
                    (setenv "LODEPNG_PATH" (string-append (assoc-ref inputs "lodepng")))
                    #t))
                (delete 'sanity-check)
                )           
        #:tests? #f            
          ))
    (inputs (list abseil-cpp eigen cmake (specification->package "pybind11@2.13.6") glfw lodepng guile-opengl python-pyopengl python-pyopengl-accelerate glm))      
    (propagated-inputs (list mujoco python-absl-py python-etils-epath python-glfw (specification->package "python-numpy@1.26.4") python-imageio))
    (native-inputs (list python-setuptools python-wheel pkg-config))
    (home-page "https://github.com/google-deepmind/mujoco/tree/main/python")
    (synopsis "MuJoCo Physics Simulator")
    (description "@code{MuJoCo} Physics Simulator.")
    (license license:asl2.0)))


(define-public python-stable-baselines3
  (package 
        (name "python-stable-baselines3")
    (version "2.7.0")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "stable_baselines3" version))
       (sha256
        (base32 "1i2y342jlrgm499shrrc7jkq585rzh4wyqj24ws55lgcbcg5cn2j"))))
    (build-system pyproject-build-system)

    (arguments
     `(#:phases
       (modify-phases %standard-phases

          (delete 'check)

         ;; (replace 'check
         ;;   (lambda* (#:key tests? #:allow-other-keys)
         ;;     (when tests?
         ;;        ; (invoke "pytest" "-v" "-m" "not expensive")
         ;;        (invoke "python3" "-m" "pytest" "-v" "-m" "not expensive"
         ;;         "--ignore=tests/test_logger.py" ; nécéssite tensorboard
         ;;         "--ignore=tests/test_save_load.py"  ;DID NOT WARN. No warnings of type (<class 'UserWarning'>,) were emitted.
         ;;         )
         ;;      )))
      )))



    (propagated-inputs (list 
                        python-cloudpickle
                        python-gymnasium-next
                        python-matplotlib
                        (specification->package "python-numpy@1.26.4")
                        python-pandas
                        python-pytorch
                        python-setuptools
                        ;; python-ale-py
                        python-tqdm
                        python-rich
                        opencv
                             ))
    (native-inputs (list python-black
                         python-mypy
                         python-pytest
                         python-pytest-cov
                         python-pytest-env
                         python-pytest-xdist
                        ;  python-ruff
                         ))
    (home-page "https://github.com/DLR-RM/stable-baselines3")
    (synopsis
     "Pytorch version of Stable Baselines, implementations of reinforcement learning algorithms.")
    (description
     "Pytorch version of Stable Baselines, implementations of reinforcement learning
algorithms.")
    (license license:expat)))


(define-public python-huggingface-sb3
  (package
    (name "python-huggingface-sb3")
    (version "3.0")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "huggingface_sb3" version))
       (sha256
        (base32 "05fmhx89i3jaxzc0r4cd0qv5b9ix8rp68a9hdsbjfn6w10c91bxb"))))
    (build-system pyproject-build-system)

    (arguments
     `(#:phases
       (modify-phases %standard-phases
          (delete 'check)
      )))

    (propagated-inputs (list python-cloudpickle python-huggingface-hub
                             (specification->package "python-numpy@1.26.4") python-pyyaml python-wasabi 
                             python-gymnasium-next python-stable-baselines3))

    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/huggingface/huggingface_sb3")
    (synopsis
     "Additional code for Stable-baselines3 to load and upload models from the Hub.")
    (description
     "Additional code for Stable-baselines3 to load and upload models from the Hub.")
    (license #f)))
(define-public python-rl-zoo3
  (package
    (name "python-rl-zoo3")
    (version "2.7.0")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/DLR-RM/rl-baselines3-zoo")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "0lbxxc7llbj6lkkzkgasp19vnm0nask54zlsf09pvg2839vafhka"))))
    (build-system pyproject-build-system)

    (arguments
     `(#:phases
       (modify-phases %standard-phases

          (delete 'check)
          )))

    (propagated-inputs (list python-gymnasium
                             python-pyyaml
                             python-rich
                             python-sb3-contrib
                             python-shimmy
                             python-tqdm))
    (native-inputs (list python-mujoco))
    (home-page "https://github.com/DLR-RM/rl-baselines3-zoo")
    (synopsis
     "A Training Framework for Stable Baselines3 Reinforcement Learning Agents")
    (description
     "This package provides a Training Framework for Stable Baselines3 Reinforcement
Learning Agents.")
    (license license:expat)))

(define-public python-sb3-contrib
  (package
    (name "python-sb3-contrib")
    (version "2.7.1")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url
              "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "111y9zdhhhmj13i0g0f3ws0z74rv31264f6sdwckl96m9jz2p26q"))))
    (build-system pyproject-build-system)

    (arguments
     `(#:phases
       (modify-phases %standard-phases

          (delete 'check)
          )))
    (propagated-inputs (list python-stable-baselines3))
    (home-page
     "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib")
    (synopsis "Contrib package of Stable Baselines3, experimental code.")
    (description "Contrib package of Stable Baselines3, experimental code.")
    (license license:expat)))

(define-public python-minatar
  (package
    (name "python-minatar")
    (version "1.0.15")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "MinAtar" version))
       (sha256
        (base32 "0fikcdbjl7ymnkxsd00lmca4lazm48qr8xgllj36w2nr55xbz6r7"))))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-cycler
                             python-kiwisolver
                             python-matplotlib
                             python-numpy
                             python-pandas
                             python-pyparsing
                             python-dateutil
                             python-pytz
                             python-scipy
                             python-seaborn
                             python-six))
        (arguments
         `(#:phases
           (modify-phases %standard-phases
              (delete 'check)
              )))
    (native-inputs (list python-setuptools-scm python-setuptools))
    (home-page "https://github.com/kenjyoung/MinAtar")
    (synopsis "A miniaturized version of the Arcade Learning Environment.")
    (description
     "This package provides a miniaturized version of the Arcade Learning Environment.")
    (license license:gpl3)))

