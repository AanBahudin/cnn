import { penjelasanSarung } from "@/utils/constants"
import { useSelector } from "react-redux"

const DetailedResult = () => {

  const {data} = useSelector((state:any) => state.globalState)

  let jenisKelamin = data?.label as string
  if (jenisKelamin.includes('Laki laki')) {
    jenisKelamin = 'Laki laki'
  } else {
    jenisKelamin = 'Perempuan'
  }

  let penjelasan = penjelasanSarung.find((item) => item.title === data?.label)


  return (
    <section className="w-full mt-6 flex flex-col gap-y-4 text-muted-foreground">

        <h3 className="text-foreground font-semibold underline">Penjelasan seputar kain</h3>
        {}
        <p className="text-sm text-muted-foreground leading-5">{penjelasan?.desc}</p>
    </section>
  )
}

export default DetailedResult