# Readme

### Environment 

- numpy == 1.18.2
- opencv == 4.3.0 

### Execution

```bash
python hw2.py [-h] [--images, -i IMAGES] [--output, -o OUTPUT]
              [--reshape_ratio, -r RESHAPE_RATIO] [--focal_length, -f FOCAL_LENGTH]
              [--crop_pixels, -c CROP_PIXELS] [--show_points, -d] [--show_pairs, -p]
```
+ -h, --help: 查看幫助
+ --images, -i: 輸入圖片的資料夾
+ --output, -o: 輸出環景圖的路徑，預設為 `full_img.png`
+ --reshape_ratio, -r: 圖片縮放比例，預設為 0.25
+ --focal_length, -f: 圖片焦距，預設為 1100
+ --crop_pixels, -c: 投到圓柱後圖片上下被減掉的高度，預設為 70
+ --show_points, -d: 輸出每張圖片的特徵點位置
+ --show_pairs, -p: 輸出相鄰兩張圖片 match 的點的位置

若要取得我們的結果，則可以直接執行以下：
```bash
python hw2.py -i pic_1/
```

