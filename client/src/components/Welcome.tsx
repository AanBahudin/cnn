import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { accordionData } from "@/utils/constants"
import ToggleTheme from "./ToggleTheme"

const Welcome = () => {
  return (
    <main className="w-1/2 h-full flex flex-col justify-start items-start p-10">
        <ToggleTheme />
        <h1 className='text-2xl font-bold mt-5'>Selamat Datang di Sistem Klasifikasi Pola Kain Tenun Tradisional Buton</h1>
        <h5 className='text-muted-foreground text-sm mt-4 leading-6'>
            Website ini dirancang untuk membantu mengenali dan mengklasifikasikan berbagai motif tradisional kain tenun khas Buton menggunakan teknologi Deep Learning. Cukup unggah gambar kain, dan sistem kami akan mendeteksi jenis pola secara otomatis — cepat, akurat, dan praktis.
        </h5>

        <Accordion type="single" collapsible className="w-full h-fit mt-4">
            {accordionData.map((item, index) => {
                return (
                    <AccordionItem key={index} value={index.toString()}>
                        <AccordionTrigger>{item.title}</AccordionTrigger>
                        <AccordionContent className="text-muted-foreground">{item.content}</AccordionContent>
                    </AccordionItem>
                )
            })}
        </Accordion>
    </main>
  )
}

export default Welcome