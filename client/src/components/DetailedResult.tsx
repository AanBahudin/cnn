import { useSelector } from "react-redux"

const DetailedResult = () => {

  const {uploadedImg, data} = useSelector((state:any) => state.globalState)

  let jenisKelamin = data?.label as string
  if (jenisKelamin.includes('Laki laki')) {
    jenisKelamin = 'Laki laki'
  } else {
    jenisKelamin = 'Perempuan'
  }

  return (
    <section className="w-full mt-6 flex flex-col gap-y-4 text-muted-foreground">
        <main className="flex w-full">
            <h5 className="w-[200px]">Termaksud Kain Buton</h5>
            <h5 className="w-[100px] text-center">:</h5>
            <h5>Ya</h5>
        </main>

        <main className="flex w-full">
            <h5 className="w-[200px]">Kategori Kain</h5>
            <h5 className="w-[100px] text-center text-muted-foreground">:</h5>
            <h5>{jenisKelamin}</h5>
        </main>

        <h3 className="capitalize text-xl my-2 text-muted-foreground font-semibold">gambar yang diunggah</h3>
        {uploadedImg && <img src={uploadedImg} className="w-[200px] bg-cover object-cover h-[150px] bg-muted rounded" />}
    </section>
  )
}

export default DetailedResult