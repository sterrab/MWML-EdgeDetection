import numpy as np
import torch

class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H1)
        self.middle_linear = torch.nn.Linear(H1, H2)
        self.output_linear = torch.nn.Linear(H2, D_out)
        self.output_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        h_relu1 = self.input_linear(x).clamp(min=0)
        h_relu2 = self.middle_linear(h_relu1).clamp(min=0)
        y_pred = self.output_linear(h_relu2)
        y_pred = self.output_softmax(y_pred)
        return y_pred

D_in, H1, H2, D_out = 5, 8, 4, 2
model = ThreeLayerNet(D_in, H1, H2, D_out)

AdamBCElr2 = 'ANNmodels/Adam.BCE.2.model.pt'
AdamBCElr4 = 'ANNmodels/Adam.BCE.4.model.pt'
AdamBCELLlr2 = 'ANNmodels/Adam.model.pt'
AdamBCELLlr4 = 'ANNmodels/Adam.BCELL.4.model.pt'

def ANNedgedetector(model_name, image, width, height):
    model.load_state_dict(torch.load(model_name))
    model.eval()

    edges_ANN_x = np.zeros((height, width))
    edges_ANN_y = np.zeros((height, width))

    for pix_y in range(2, int(height-2)):
        for pix_x in range(2, int(width-2)):
            inputs_x = torch.from_numpy(image[pix_y, pix_x-2:pix_x+3].reshape((1, 5))/max(abs(image[pix_y, pix_x-2:pix_x+3])))
            inputs_y = torch.from_numpy(image[pix_y-2:pix_y+3, pix_x].reshape((1, 5))/max(abs(image[pix_y-2:pix_y+3, pix_x])))
            model_output_x = torch.argmax(model(inputs_x.float()), dim=1)
            model_output_y = torch.argmax(model(inputs_y.float()), dim=1)

            if model_output_x == torch.tensor([1]):
                edges_ANN_x[pix_y, pix_x] = 255
            elif model_output_y == torch.tensor([1]):
                edges_ANN_y[pix_y, pix_x] = 255

    return edges_ANN_y, edges_ANN_x



