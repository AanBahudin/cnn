import { useSelector } from 'react-redux'

const ProcessingDataContainer = () => {

    const {processingData} = useSelector((state:any) => state.globalState)
    if (processingData.data === 'tested') {
        return (
            <section className="w-1/2 h-full flex flex-col justify-start items-start p-10">
                <h1 className='text-4xl font-semibold text-center'>Belum ada hasil</h1>
            </section>
        )
    }

    const base64Image = processingData?.processed_image_base64;

    return (
        <section className="w-1/2 h-full flex flex-col justify-start items-start p-10">
            <img src={`data:image/jpeg;base64,${base64Image}`} className="w-[300px] bg-fill bg-primary object-cover h-[300px] rounded" />
        </section>
    )
}

export default ProcessingDataContainer