import picture1 from '@/assets/images/examples/picture (1).png'
import picture2 from '@/assets/images/examples/picture (2).png'
import picture3 from '@/assets/images/examples/picture (3).png'
import picture4 from '@/assets/images/examples/picture (4).png'
import picture5 from '@/assets/images/examples/picture (5).png'
import picture6 from '@/assets/images/examples/picture (6).png'
import picture7 from '@/assets/images/examples/picture (7).png'
import picture8 from '@/assets/images/examples/picture (8).png'

const UnknownResult = () => {

    const examples = [picture1, picture2, picture3, picture4, picture5, picture6, picture7, picture8]

  return (
     <section className="w-full flex flex-col gap-y-2 text-muted-foreground">
        <h3 className="text-lg mt-4 text-foreground font-semibold">Perhatian</h3>
        <p className="text-sm text-muted-foreground leading-5">Motif kain pada gambar tidak dikenali sebagai salah satu dari delapan kategori kain tenun tradisional yang didukung oleh sistem.</p>

        <h3 className="text-lg mb-4 text-foreground font-semibold">Contoh gambar yang dikenali oleh sistem</h3>
        <main className="w-fit grid grid-cols-4 gap-x-10 gap-y-4 justify-items-start">
            {examples.map((item: any, index: number) => {
            return (
                <img key={index} src={item} className="w-[100px] h-[100px] object-cover rounded border bg-muted" alt="Gambar serupa" />
            )
            })}
        </main>
   </section>
  )
}

export default UnknownResult