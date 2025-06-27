import { cn } from '@/lib/utils'
import React from 'react'

const Container = ({children, className} : {children: React.ReactNode, className?: string}) => {
  return (
    <section className={cn('w-[90%] mx-auto h-fit rounded-xl p-10', className)}>
        {children}
    </section>
  )
}

export default Container