import robbielib as robbie

print(robbie.add(1,6))

network = robbie.Network()
network.loss_tol = 5e-4;
print(network.loss_tol)

layer = robbie.FCLayer(2,3)
print( layer.name() )