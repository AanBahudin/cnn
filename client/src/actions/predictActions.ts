import { customFetch } from "@/utils/customFetch";
import { store } from "@/store";
import { removeImage, setData, setUploaded } from "@/cart/globalSlice";

export const predictModel = async(formData: FormData) => {

  const uploadedImage = formData.get('motifKain') as File
  const previewUrl = URL.createObjectURL(uploadedImage);

  const response = await customFetch.post('/data/processing', formData)  
  if (response.status >= 400) {
      return {message: 'Terjadi Kesalahan', deskripsi: 'kesalahan dalam mengupload gambar. Silahkan coba lagi nanti'}
  }

  console.log(response.data)

  // gambar yang ditampilan di preview gambar sebelum dikirim ke model
  store.dispatch(setData(response.data))
  store.dispatch(removeImage())

  // gambar yang ditampilkan di hasil klasifikasi
  store.dispatch(setUploaded(previewUrl))

  return {message : 'Prediksi Selesai'}
}