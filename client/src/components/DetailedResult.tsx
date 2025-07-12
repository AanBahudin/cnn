import { penjelasanSarung } from "@/utils/constants"
import { useSelector } from "react-redux"

const DetailedResult = () => {

  const {data} = useSelector((state:any) => state.globalState)
  let penjelasan = penjelasanSarung.find((item) => item.title === data?.label)

  return (
    <section className="w-full mt-6 flex flex-col gap-y-4 text-muted-foreground">
      <h3 className="text-foreground font-semibold underline">Penjelasan seputar kain</h3>
      <p className="text-sm text-muted-foreground leading-5">{penjelasan?.desc}</p>


      <h3 className="text-foreground font-semibold underline">Gambar serupa</h3>
      <main className="flex items-center justify-start gap-x-6">
        {data?.examples.map((item: any, index: number) => {
          return (
            <img key={index} src={item} className="w-[100px] h-[120px] object-contain rounded border bg-muted" alt="Gambar serupa" />
          )
        })}
      </main>
    </section>
  )
}

export default DetailedResult