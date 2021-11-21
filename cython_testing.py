from geometry import SphereTree, Scene, Envelope
import time

if True:
    # Testing for 'Check collision cythonization'

    # Scene Setup
    file = 'meshes/GenerativeBracket.stl'
    obj1 = SphereTree(file, 10)
    obj1.translate([380.0 / 2.5, 284.0 / 4 + 30, 60 + 20])

    obj2 = SphereTree(file, 10)
    obj2.rotate([90, 0, 0])
    obj2.rotate([45, 90, 0])
    obj2.translate([380.0 / 2, 284.0 / 2, 380.0 / 2])

    obj3 = SphereTree(file, 10)
    obj3.rotate(obj2.total_rotation_matrix, matrix=True)
    obj3.translate([380.0 / 3.2, 284.0 / 1.5, 380.0 / 2 + 5])

    scene = Scene(part_interval=1.5)
    scene.add_parts([obj1, obj2, obj3])
    scene.add_envelope(Envelope(380, 284, 380))
    #scene.visualize()

    # Prepare implementation testing
    ids = list(scene.parts.keys())
    pairs = [[ids[0], ids[1]],
             [ids[0], ids[2]],
              [ids[1], ids[2]]]

    # Number of test loops
    n = 10

    cy = 0
    c1 = time.time()
    while cy < n:
        for pairing in pairs:
            scene.check_collision(pairing, cy=True)
        cy += 1
    c2 = time.time()
    cytime = c2-c1
    print('Cython testing done: {0}'.format(cytime))

    py = 0
    p1 = time.time()
    while py < n:
        for pairing in pairs:
            scene.check_collision(pairing)
        py += 1
    p2 = time.time()
    pytime = p2-p1
    print('Python testing done: {0}'.format(pytime))

    print('Cython is {0}x faster than Python over {1} loops.'.format(round(pytime/cytime,2), n))


