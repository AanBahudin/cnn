import DetailedResult from "./DetailedResult"
import Welcome from "./Welcome"
import Loading from "./Loading"
import { useFormStatus } from "@/context/FormContext"
import { useSelector } from "react-redux"
import UnknownResult from "./UnknownResult"

const DataContainer = () => {

  const {isLoading} = useFormStatus()
  const {data} = useSelector((state:any) => state.globalState)

  if (data.type === 'tested') {
    return <Welcome />
  }

  if (isLoading) {
    return <Loading />
  }


  let title : string = data?.label

  if (data?.status === 'recognized') {
    if (title.includes('LK')) {
      title = title.replace('LK', 'Laki laki')
    } else {
      title = title.replace('P', 'Perempuan')
    }
  
  
    if (title.includes('Katamba Layana')) {
      title = title.replace('Katamba Layana', 'Katamba Gawu')
    }
  }

  
  return (
    <main className="w-3/5 h-full flex flex-col justify-start items-start p-6">
        <h3 className="text-lg uppercase font-semibold bg-primary rounded px-6 text-white">hasil klasifikasi</h3>
        <h1 className=" text-3xl font-bold uppercase mt-3">
          {data?.status === 'recognized' ? (
            `Motif Kain ${title}`
          ) : (
            'Motif Kain Tidak Dikenali'
          )}
        </h1>

        {data?.status === 'recognized' ? <DetailedResult /> : <UnknownResult />}
        
    </main>
  )
}

export default DataContainer