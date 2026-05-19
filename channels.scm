(use-modules (ice-9 match))

(define script-dir (dirname (current-filename)))
(define relative-path "guix-rl")
(define full-path (string-join (list script-dir relative-path) "/"))

(list (channel
        (name 'guix-rl)
        (url full-path)
        (branch "main")
        (commit
          "2767f9d6025a203873d2cc20193f3be212f79a9d"))
      (channel
        (name 'guix)
        (url "https://git.guix.gnu.org/guix.git")
        (branch "master")
        (commit
          "46e1b73b449c526b987aa2872cdbeec91403b49e")
        (introduction
          (make-channel-introduction
            "9edb3f66fd807b096b48283debdcddccfea34bad"
            (openpgp-fingerprint
              "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
)
