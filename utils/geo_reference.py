from osgeo import osr, gdal
import glob
import os

def assign_spatial_reference_byfile(src_path, dst_path):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds = None
    src_ds = None


def def_geoCoordSys(read_path, img_transf, img_proj):
    array_dataset = gdal.Open(read_path)
    img_array = array_dataset.ReadAsArray(0, 0, array_dataset.RasterXSize, array_dataset.RasterYSize)
    if 'int8' in img_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img_array.shape) == 3:
        img_bands, im_height, im_width = img_array.shape
    else:
        img_bands, (im_height, im_width) = 1, img_array.shape

    filename = read_path[:-4] + '_proj' + read_path[-4:]
    driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
    dataset = driver.Create(filename, im_width, im_height, img_bands, datatype)
    dataset.SetGeoTransform(img_transf)  # 写入仿射变换参数
    dataset.SetProjection(img_proj)  # 写入投影

    # 写入影像数据
    if img_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img_array)
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img_array[i])
    print(read_path, 'geoCoordSys get!')




if __name__ == "__main__":
    src_root = r"E:\chenximing\Cracks\GoafCrack_v6\datuxiang\kq7"
    dst_root = r"E:\chenximing\Cracks\GoafCrack_v6\datuxiang\kq7_mask/"

    src_paths = glob.glob(os.path.join(src_root, "*.tif"))
    dst_paths = glob.glob(os.path.join(dst_root, "*.png"))
    print(src_paths)
    dst_paths = sorted(dst_paths, key=lambda x: len(x))
    print(dst_paths)

    for src_path, dst_path in zip(src_paths, dst_paths):
        src_dataset = gdal.Open(src_path)
        src_pos_transf = src_dataset.GetGeoTransform()
        src_pos_proj = src_dataset.GetProjection()
        def_geoCoordSys(dst_path, src_pos_transf, src_pos_proj)