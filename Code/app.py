from astropy.io import fits
from astropy.table import Table
import numpy as np

# 生成FLUX和WAVELENGTH数组
flux = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
wavelength = np.array([500, 600, 700, 800, 900])

# 创建一个Table对象，并将FLUX和WAVELENGTH作为列添加到该对象中
table = Table([flux, wavelength], names=['FLUX', 'WAVELENGTH'])

# 创建一个新的HDUList对象
hdulist = fits.HDUList()

# 将Table对象转换为BinTableHDU对象，并将其添加到HDUList中
table_hdu = fits.BinTableHDU(table)
hdulist.append(table_hdu)

# 正确写法（写入具体文件）
output_path = r"E:\CNN-Image-Segmentation-master\output.fits"  # 使用原始字符串(r)避免转义问题
hdulist.writeto(output_path, overwrite=True)

