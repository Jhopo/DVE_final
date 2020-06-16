# DVE_final

## Requirement
```
numpy
scipy
opencv-python
```

## Usage
Example:
```
python main.py --filename example --ext jpg --output_dir output --k 3 --save --rgb

```
Required arguments:
```
main.py

--filename    圖片檔名
--ext         圖片附檔名
--output_dir  輸出資料夾

--rgb         是否用全彩模式，反之則用灰階
--save        是否儲存結果圖片

--k           找local extrema時用的 k*k kernel，預設為 3
--scale       圖片縮放比例，預設為 1
```
完成的圖片放在`output_dir`
包括local extrema, max/min envelope, difference, mean
