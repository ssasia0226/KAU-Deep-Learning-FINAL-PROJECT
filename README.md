# KAU deep learning FINAL PROJECT


## how to run your source code 

  
작업환경 : Google Colab, Jupyter Notebook(GPU)

실행시 주의사항

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

=> gpu개수에 따라 num_workers를 제거하거나 수정한다.


코드에 datasets를 받는 코드가 있기때문에 따로 다운받지 않아도 된다. (dataset코드 홈페이지를 보고서에 첨부하였다.)
