import torch


class Tnet(torch.nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k = k
      self.conv1 = torch.nn.Conv1d(k, 64, 1)
      self.conv2 = torch.nn.Conv1d(64, 128, 1)
      self.conv3 = torch.nn.Conv1d(128, 1024, 1)
      self.fc1 = torch.nn.Linear(1024, 512)
      self.fc2 = torch.nn.Linear(512, 256)
      self.fc3 = torch.nn.Linear(256, k * k)

      self.bn1 = torch.nn.BatchNorm1d(64)
      self.bn2 = torch.nn.BatchNorm1d(128)
      self.bn3 = torch.nn.BatchNorm1d(1024)
      self.bn4 = torch.nn.BatchNorm1d(512)
      self.bn5 = torch.nn.BatchNorm1d(256)
       
   def forward(self, input):
      bs = input.size(0)
      xb = torch.nn.functional.relu(self.bn1(self.conv1(input)))
      xb = torch.nn.functional.relu(self.bn2(self.conv2(xb)))
      xb = torch.nn.functional.relu(self.bn3(self.conv3(xb)))
      pool = torch.nn.MaxPool1d(xb.size(-1))(xb)
      flat = torch.nn.Flatten(1)(pool)
      xb = torch.nn.functional.relu(self.bn4(self.fc1(flat)))
      xb = torch.nn.functional.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1, self.k, self.k) + init
      return matrix


class Transform(torch.nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
       

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        xb = torch.nn.functional.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = torch.nn.functional.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = torch.nn.MaxPool1d(xb.size(-1))(xb)
        output = torch.nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64


class PointNet(torch.nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, classes)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(p=.3)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = torch.nn.functional.relu(self.bn1(self.fc1(xb)))
        xb = torch.nn.functional.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64
    

if __name__ == '__main__':
    model = PointNet()
    print('Num params: ', sum(p.numel() for p in model.parameters()))
    test_x = torch.rand(1, 3, 1024)
    test_x = test_x.transpose(1, 2)
    test_y = model(test_x)
    print('Output shape: ', test_y.shape)
