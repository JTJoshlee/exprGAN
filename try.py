def Grad_CAM(model, dataset):

    target_layer = [model.encoder.layers[-1]]
    neutral_name = dataset['neutral_name']
    smile_name = dataset['smile_name']
    data_num = int(dataset['neutral'].shape[0])
    
    for i in range(data_num): 
        print(dataset['neutral'][i])
        neutral_tensor = dataset['neutral'][i].unsqueeze(0)
        print(f"neutral shape:{neutral_tensor.shape}")
        smile_tensor = dataset['smile'][i].unsqueeze(0)
        cam = GradCAM(model=model, target_layers=target_layer)
        neutral_targets = [ClassifierOutputTarget(0)]
        smile_targets = [ClassifierOutputTarget(1)]
        grayscale_cam_neutral = cam(input_tensor=neutral_tensor, targets=neutral_targets)
        grayscale_cam_smile = cam(input_tensor=smile_tensor, targets=smile_targets)
            

        neutral_cam_images = []
        
        print(f'grayscale_cam :{grayscale_cam_neutral}')
        neutral_img_np = neutral_tensor.squeeze(0).cpu().detach().numpy()  # 移除 batch 维度
        print(f"neutral_img_np:{neutral_img_np}")
        neutral_img_np = np.transpose(neutral_img_np, (1, 2, 0))
        # neutral_img_np = np.transpose(neutral_img_np, (1, 2, 0))  # 转换为 (H, W, C)
        smile_img_np = smile_tensor.squeeze(0).cpu().detach().numpy()  # 同样处理 smile tensor
        smile_img_np = np.transpose(smile_img_np, (1, 2, 0)) 
            # 将 Grad-CAM 映射到原始图像
        neutral_cam_image = show_cam_on_image(neutral_img_np, grayscale_cam_neutral, use_rgb=True)
        #smile_cam_image = show_cam_on_image(smile_img_np, grayscale_cam_smile, use_rgb=True)   
            # 保存生成的 CAM 图像
        cv2.imwrite(f"E:/style_exprGAN/ORL_data/choosed/neutral_cam/CAM_test_{dataset['neutral_name'][i]}.jpg", neutral_cam_image)  # 使用不同的文件名
        cv2.imwrite(f"E:/style_exprGAN/ORL_data/choosed/smile_cam/CAM_test_{dataset['smile_name'][i]}.jpg", smile_cam_image)  # 使用不同的文件名
        #cam_images.append(cam_image)  # 将生成的 CAM 图像添加到列表

    #return cam_images

    